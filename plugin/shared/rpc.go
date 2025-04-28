package shared

import (
	"ner-backend/plugin/proto"
	"net/rpc"
)

// RPCClient is an implementation of KV that talks over RPC.
type RPCClient struct{ client *rpc.Client }

func (m *RPCClient) Predict(sentence string) ([]*proto.Entity, error) {
	var resp []*proto.Entity
	err := m.client.Call("Plugin.Predict", sentence, &resp)
	return resp, err
}

// Here is the RPC server that RPCClient talks to, conforming to
// the requirements of net/rpc
type RPCServer struct {
	// This is the real implementation
	Impl Model
}

func (m *RPCServer) Predict(sentence string, resp *[]*proto.Entity) error {
	v, err := m.Impl.Predict(sentence)
	*resp = v
	return err
}
