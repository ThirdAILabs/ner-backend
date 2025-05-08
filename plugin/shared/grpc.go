package shared

import (
	"context"

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

func (m *GRPCClient) PredictBatch(sentences []string) ([]*proto.PredictResponse, error) {
	resp, err := m.client.PredictBatch(context.Background(), &proto.PredictBatchRequest{
		Sentences: sentences,
	})
	if err != nil {
		return nil, err
	}

	return resp.Predictions, nil
}

// Here is the gRPC server that GRPCClient talks to.
type GRPCServer struct {
	// This is the real implementation
	Impl Model
}

func (m *GRPCServer) Predict(
	ctx context.Context,
	req *proto.PredictRequest) (*proto.PredictResponse, error) {
	v, err := m.Impl.Predict(req.Sentence)
	return &proto.PredictResponse{Entities: v}, err
}

func (m *GRPCServer) PredictBatch(
	ctx context.Context,
	req *proto.PredictBatchRequest) (*proto.PredictBatchResponse, error) {
	v, err := m.Impl.PredictBatch(req.Sentences)
	return &proto.PredictBatchResponse{Predictions: v}, err
}
