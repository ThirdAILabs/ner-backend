// Package shared contains shared data between the host and plugins.
package shared

import (
	"context"

	"google.golang.org/grpc"

	"ner-backend/plugin/proto"

	"github.com/hashicorp/go-plugin"
)

// Handshake is a common handshake that is shared by plugin and host.
var Handshake = plugin.HandshakeConfig{
	// This isn't required when using VersionedPlugins
	ProtocolVersion:  1,
	MagicCookieKey:   "BASIC_PLUGIN",
	MagicCookieValue: "hello",
}

// PluginMap is the map of plugins we can dispense.
var PluginMap = map[string]plugin.Plugin{
	"model_grpc": &ModelGRPCPlugin{},
}

// KV is the interface that we're exposing as a plugin.
type Model interface {
	Predict(sentence string) ([]*proto.Entity, error)
	PredictBatch(sentences []string) ([]*proto.PredictResponse, error)
	Finetune(prompt string, tags []*proto.TagInfo, samples []*proto.Sample) error
	Save(dir string) error
}

// This is the implementation of plugin.GRPCPlugin so we can serve/consume this.
type ModelGRPCPlugin struct {
	// GRPCPlugin must still implement the Plugin interface
	plugin.Plugin
	// Concrete implementation, written in Go. This is only used for plugins
	// that are written in Go.
	Impl Model
}

func (p *ModelGRPCPlugin) GRPCServer(broker *plugin.GRPCBroker, s *grpc.Server) error {
	proto.RegisterModelServer(s, &GRPCServer{Impl: p.Impl})
	return nil
}

func (p *ModelGRPCPlugin) GRPCClient(ctx context.Context, broker *plugin.GRPCBroker, c *grpc.ClientConn) (interface{}, error) {
	return &GRPCClient{client: proto.NewModelClient(c)}, nil
}
