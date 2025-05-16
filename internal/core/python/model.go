package python

import (
	"fmt"
	"ner-backend/internal/core/types"
	"ner-backend/internal/core/utils"
	"ner-backend/pkg/api"
	"ner-backend/plugin/proto"
	"ner-backend/plugin/shared"
	"os/exec"

	"github.com/hashicorp/go-plugin"
)

// TODO: this object is not thread-safe, implement a mutex to protect
// concurrent access to the plugin client APIs
type PythonModel struct {
	client *plugin.Client
	model  shared.Model
}

func LoadPythonModel(PythonExecutable, PluginScript, PluginModelName, KwargsJSON string) (*PythonModel, error) {
	client := plugin.NewClient(&plugin.ClientConfig{
		HandshakeConfig: shared.Handshake,
		Plugins:         shared.PluginMap,
		Cmd: exec.Command(
			PythonExecutable,
			PluginScript,
			"--model-name", PluginModelName,
			"--model-config", KwargsJSON,
		),
		AllowedProtocols: []plugin.Protocol{
			plugin.ProtocolNetRPC, plugin.ProtocolGRPC},
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

func (ner *PythonModel) Finetune(prompt string, tags []api.TagInfo, samples []api.Sample) error {
	const maxPayload = 2 * 1024 * 1024 // 2 MB

	// convert TagInfo
	protoTags := make([]*proto.TagInfo, len(tags))
	for i, t := range tags {
		protoTags[i] = &proto.TagInfo{
			Name:        t.Name,
			Description: t.Description,
			Examples:    t.Examples,
		}
	}

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
			)
			allEntities = append(allEntities, ents...)
		}

		i = j
	}

	return allEntities, nil
}

func convertProtoEntitiesToTypes(protoEntities []*proto.Entity, text string, startOffset int) []types.Entity {

	typesEntities := make([]types.Entity, len(protoEntities))
	runes := []rune(text)
	for i, pe := range protoEntities {
		if pe != nil {
			start := int(pe.Start) + startOffset
			end := int(pe.End) + startOffset
			typesEntities[i] = types.CreateEntityWithRune(pe.Label, runes, start, end)
		}
	}
	return typesEntities
}
