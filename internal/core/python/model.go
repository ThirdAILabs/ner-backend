package python

import (
	"fmt"
	"ner-backend/internal/core/types"
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

func (ner *PythonModel) Predict(text string) ([]types.Entity, error) {
	result, err := ner.model.Predict(text)
	if err != nil {
		return nil, err
	}

	return convertProtoEntitiesToTypes(result), nil
}

func (ner *PythonModel) Finetune(taskPrompt string, tags []api.TagInfo, samples []api.Sample) error {
	return fmt.Errorf("Finetune not implemented")
}

func (ner *PythonModel) Save(path string) error {
	return fmt.Errorf("save not implemented")
}

func (ner *PythonModel) Release() {
	if ner.client == nil {
		return
	}

	ner.client.Kill()
	ner.client = nil
	ner.model = nil
}

func convertProtoEntitiesToTypes(protoEntities []*proto.Entity) []types.Entity {
	typesEntities := make([]types.Entity, len(protoEntities))

	for i, pe := range protoEntities {
		if pe != nil {
			typesEntities[i].Label = pe.Label
			typesEntities[i].Text = pe.Text
			typesEntities[i].Start = int(pe.Start)
			typesEntities[i].End = int(pe.End)
		}
	}

	return typesEntities
}
