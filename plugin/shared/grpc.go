// Copyright (c) HashiCorp, Inc.
// SPDX-License-Identifier: MPL-2.0

package shared

import (
	"context"

	"ner-backend/plugin/proto"
)

// GRPCClient is an implementation of KV that talks over RPC.
type GRPCClient struct{ client proto.ModelClient }

func (m *GRPCClient) Predict(sentence string) ([]byte, error) {
	resp, err := m.client.Predict(context.Background(), &proto.PredictRequest{
		Sentence: sentence,
	})
	if err != nil {
		return nil, err
	}

	return resp.Value, nil
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
	return &proto.PredictResponse{Value: v}, err
}
