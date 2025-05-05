package shared

import (
	"context"
	"fmt"

	"ner-backend/plugin/proto"
)

// GRPCClient is an implementation of KV that talks over RPC.
type GRPCClient struct{ client proto.ModelClient }

func (m *GRPCClient) Predict(sentence string) ([]*proto.Entity, error) {
	resp, err := m.client.Predict(context.Background(), &proto.PredictRequest{
		Sentence: sentence,
	})
	if err != nil {
		return nil, err
	}

	return resp.Entities, nil
}

// Here is the gRPC server that GRPCClient talks to.
type GRPCServer struct {
	// This is the real implementation
	Impl Model
}

func (m *GRPCServer) Predict(
	ctx context.Context,
	req *proto.PredictRequest) (*proto.PredictResponse, error) {
	fmt.Println("GRPCServer Predict called, req:", req)
	v, err := m.Impl.Predict(req.Sentence)
	if err != nil {
		fmt.Println("GRPCServer Predict error:", err)
		return nil, err
	}
	fmt.Println("GRPCServer Predict response:", v)
	return &proto.PredictResponse{Entities: v}, err
}
