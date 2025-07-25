package python

import (
	"fmt"
	"log/slog"
	"math/rand/v2"
	"ner-backend/internal/core/types"
	"ner-backend/internal/core/utils"
	"ner-backend/pkg/api"
	"ner-backend/plugin/proto"
	"ner-backend/plugin/shared"
	"os"
	"os/exec"
	"time"

	"github.com/hashicorp/go-plugin"
)

var (
	pythonExecutable string
	pluginScript     string
)

func EnablePythonPlugin(_pythonExecutable, _pluginScript string) {
	pythonExecutable = _pythonExecutable
	pluginScript = _pluginScript
}

func PythonPluginEnabled() bool {
	return pythonExecutable != "" && pluginScript != ""
}

// TODO: this object is not thread-safe, implement a mutex to protect
// concurrent access to the plugin client APIs
type PythonModel struct {
	client *plugin.Client
	model  shared.Model
}

func LoadPythonModel(pluginModelName, kwargsJSON string) (*PythonModel, error) {
	var cmd *exec.Cmd

	// If PLUGIN_SERVER env var is set, use the PyInstaller executable
	if pluginServer := os.Getenv("PLUGIN_SERVER"); pluginServer != "" {
		cmd = exec.Command(pluginServer, "--model-name", pluginModelName, "--model-config", kwargsJSON)
	} else {
		if !PythonPluginEnabled() {
			return nil, fmt.Errorf("python is not enabled for finetuning or inference on this model")
		}
		// Fallback to using Python interpreter + script for development
		cmd = exec.Command(pythonExecutable, pluginScript, "--model-name", pluginModelName, "--model-config", kwargsJSON)
	}

	client := plugin.NewClient(&plugin.ClientConfig{
		HandshakeConfig: shared.Handshake,
		Plugins:         shared.PluginMap,
		Cmd:             cmd,
		AllowedProtocols: []plugin.Protocol{
			plugin.ProtocolNetRPC, plugin.ProtocolGRPC},
		StartTimeout: 5 * time.Minute,
	})

	rpcClient, err := client.Client()
	if err != nil {
		return nil, fmt.Errorf("error establishing RPC connection: %w", err)
	}

	raw, err := rpcClient.Dispense("model_grpc")
	if err != nil {
		client.Kill()
		return nil, fmt.Errorf("error dispensing '%s': %w", "model_grpc", err)
	}

	model, ok := raw.(shared.Model)
	if !ok {
		client.Kill()
		return nil, fmt.Errorf("dispensed interface '%s' is not of expected type shared.Model (actual type: %T)", "model_grpc", raw)
	}

	return &PythonModel{
		client: client,
		model:  model,
	}, nil
}

func LoadCnnModel(modelDir, keyB64 string) (*PythonModel, error) {
	cfgJSON := fmt.Sprintf(`{"model_path":"%s", "tokenizer_path":"%s/qwen_tokenizer", "key_b64": "%s"}`, modelDir, modelDir, keyB64)
	return LoadPythonModel("python_cnn_ner_model", cfgJSON)
}

func (ner *PythonModel) FinetuneAndSave(prompt string, tags []types.TagInfo, samples []api.Sample, savePath string) error {
	const maxPayload = 2 * 1024 * 1024 // 2 MB

	// convert TagInfo
	protoTags := make([]*proto.TagInfo, len(tags))
	for i, t := range tags {
		protoTags[i] = &proto.TagInfo{
			Name:        t.Name,
			Description: t.Desc,
			Examples:    t.Examples,
		}
	}
	for epoch := 0; epoch < 5; epoch++ {
		slog.Info("finetuning epoch", "epoch", epoch)

		// shuffle samples each epoch
		rand.Shuffle(len(samples), func(i, j int) {
			samples[i], samples[j] = samples[j], samples[i]
		})
		type chunk struct {
			samples []*proto.Sample
			size    int
		}
		var curr chunk

		flush := func() error {
			if len(curr.samples) == 0 {
				return nil
			}
			if err := ner.model.Finetune(prompt, protoTags, curr.samples); err != nil {
				return fmt.Errorf("finetune chunk error: %w", err)
			}
			curr.samples = nil
			curr.size = 0
			return nil
		}
		for _, s := range samples {
			p := &proto.Sample{
				Tokens: s.Tokens,
				Labels: s.Labels,
			}
			est := 0
			for _, tok := range p.Tokens {
				est += len(tok)
			}
			for _, lab := range p.Labels {
				est += len(lab)
			}
			if curr.size+est > maxPayload {
				if err := flush(); err != nil {
					return fmt.Errorf("finetune chunk error: %w", err)
				}
			}
			curr.samples = append(curr.samples, p)
			curr.size += est
		}
		if err := flush(); err != nil {
			return fmt.Errorf("final finetune chunk error: %w", err)
		}
	}

	if err := ner.Save(savePath); err != nil {
		return fmt.Errorf("error saving model: %w", err)
	}

	return nil
}

func (ner *PythonModel) Save(path string) error {
	return ner.model.Save(path)
}

func (ner *PythonModel) Release() {
	if ner.client == nil {
		return
	}

	ner.client.Kill()
	ner.client = nil
	ner.model = nil
}

func (ner *PythonModel) Predict(text string) ([]types.Entity, error) {
	sentences, startOffsets := utils.SplitText(text)

	// we will send max size of 2 MB at once
	const maxPayload = 2 * 1024 * 1024 // 2 MB

	var allEntities []types.Entity
	for i := 0; i < len(sentences); {
		var (
			batchBytes int
			j          = i
		)
		// we accumulate sentences until we hit the payload
		for ; j < len(sentences); j++ {
			batchBytes += len(sentences[j])
			if batchBytes > maxPayload {
				break
			}
		}
		// we ensure atleast one sentence per payload
		if j == i {
			j = i + 1
		}

		batchResults, err := ner.model.PredictBatch(sentences[i:j])
		if err != nil {
			return nil, err
		}

		// we convert each sub-slice’s entities back into global offsets
		for k, resp := range batchResults {
			idx := i + k
			ents := convertProtoEntitiesToTypes(
				resp.Entities,
				text,
				startOffsets[idx],
				sentences[idx],
			)
			allEntities = append(allEntities, ents...)
		}

		i = j
	}

	return allEntities, nil
}

func convertProtoEntitiesToTypes(protoEntities []*proto.Entity, text string, sentenceOffset int, sentence string) []types.Entity {
	if len(protoEntities) == 0 {
		return nil // so we can skip unicode offset computation if no entities
	}

	// Python returns offsets in terms of unicode code points, this converts that back into a byte offset.
	runeIdxToByteIdx := make([]int, len(sentence)+1)
	cnt := 0
	for i := range sentence {
		runeIdxToByteIdx[cnt] = i
		cnt++
	}
	runeIdxToByteIdx[cnt] = len(sentence)
	runeIdxToByteIdx = runeIdxToByteIdx[:cnt+1]

	typesEntities := make([]types.Entity, len(protoEntities))
	for i, pe := range protoEntities {
		if pe != nil {
			start := runeIdxToByteIdx[pe.Start] + sentenceOffset
			end := runeIdxToByteIdx[pe.End] + sentenceOffset
			typesEntities[i] = types.CreateEntity(pe.Label, text, start, end)
		}
	}
	return typesEntities
}
