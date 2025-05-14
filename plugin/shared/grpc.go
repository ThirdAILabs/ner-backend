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

func (m *GRPCClient) Finetune(prompt string, tags []*proto.TagInfo, samples []*proto.Sample) error {
	resp, err := m.client.Finetune(context.Background(), &proto.FinetuneRequest{
		Prompt:  prompt,
		Tags:    tags,
		Samples: samples,
	})
	if err != nil {
		return err
	}
	if !resp.Success {
		return err
	}
	return nil
}

func (m *GRPCClient) Save(dir string) error {
	resp, err := m.client.Save(context.Background(), &proto.SaveRequest{
		Dir: dir,
	})
	if err != nil {
		return err
	}

	if !resp.Success {
		return err
	}
	return nil
}

// Here is the gRPC server that GRPCClient talks to.
type GRPCServer struct {
	proto.UnimplementedModelServer
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

func (m *GRPCServer) Finetune(
	ctx context.Context,
	req *proto.FinetuneRequest,
) (*proto.FinetuneResponse, error) {
	err := m.Impl.Finetune(req.Prompt, req.Tags, req.Samples)
	if err != nil {
		return &proto.FinetuneResponse{Success: false}, err
	}
	return &proto.FinetuneResponse{Success: true}, nil
}

func (m *GRPCServer) Save(
	ctx context.Context,
	req *proto.SaveRequest,
) (*proto.SaveResponse, error) {
	err := m.Impl.Save(req.Dir)
	if err != nil {
		return &proto.SaveResponse{Success: false}, err
	}
	return &proto.SaveResponse{Success: true}, nil
}
