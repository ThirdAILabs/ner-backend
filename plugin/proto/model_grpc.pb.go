// Code generated by protoc-gen-go-grpc. DO NOT EDIT.
// versions:
// - protoc-gen-go-grpc v1.3.0
// - protoc             (unknown)
// source: proto/model.proto

package proto

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
)

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
// Requires gRPC-Go v1.32.0 or later.
const _ = grpc.SupportPackageIsVersion7

const (
	Model_Predict_FullMethodName      = "/proto.Model/Predict"
	Model_PredictBatch_FullMethodName = "/proto.Model/PredictBatch"
	Model_Finetune_FullMethodName     = "/proto.Model/Finetune"
	Model_Save_FullMethodName         = "/proto.Model/Save"
)

// ModelClient is the client API for Model service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://pkg.go.dev/google.golang.org/grpc/?tab=doc#ClientConn.NewStream.
type ModelClient interface {
	Predict(ctx context.Context, in *PredictRequest, opts ...grpc.CallOption) (*PredictResponse, error)
	PredictBatch(ctx context.Context, in *PredictBatchRequest, opts ...grpc.CallOption) (*PredictBatchResponse, error)
	Finetune(ctx context.Context, in *FinetuneRequest, opts ...grpc.CallOption) (*FinetuneResponse, error)
	Save(ctx context.Context, in *SaveRequest, opts ...grpc.CallOption) (*SaveResponse, error)
}

type modelClient struct {
	cc grpc.ClientConnInterface
}

func NewModelClient(cc grpc.ClientConnInterface) ModelClient {
	return &modelClient{cc}
}

func (c *modelClient) Predict(ctx context.Context, in *PredictRequest, opts ...grpc.CallOption) (*PredictResponse, error) {
	out := new(PredictResponse)
	err := c.cc.Invoke(ctx, Model_Predict_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelClient) PredictBatch(ctx context.Context, in *PredictBatchRequest, opts ...grpc.CallOption) (*PredictBatchResponse, error) {
	out := new(PredictBatchResponse)
	err := c.cc.Invoke(ctx, Model_PredictBatch_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelClient) Finetune(ctx context.Context, in *FinetuneRequest, opts ...grpc.CallOption) (*FinetuneResponse, error) {
	out := new(FinetuneResponse)
	err := c.cc.Invoke(ctx, Model_Finetune_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *modelClient) Save(ctx context.Context, in *SaveRequest, opts ...grpc.CallOption) (*SaveResponse, error) {
	out := new(SaveResponse)
	err := c.cc.Invoke(ctx, Model_Save_FullMethodName, in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// ModelServer is the server API for Model service.
// All implementations should embed UnimplementedModelServer
// for forward compatibility
type ModelServer interface {
	Predict(context.Context, *PredictRequest) (*PredictResponse, error)
	PredictBatch(context.Context, *PredictBatchRequest) (*PredictBatchResponse, error)
	Finetune(context.Context, *FinetuneRequest) (*FinetuneResponse, error)
	Save(context.Context, *SaveRequest) (*SaveResponse, error)
}

// UnimplementedModelServer should be embedded to have forward compatible implementations.
type UnimplementedModelServer struct {
}

func (UnimplementedModelServer) Predict(context.Context, *PredictRequest) (*PredictResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Predict not implemented")
}
func (UnimplementedModelServer) PredictBatch(context.Context, *PredictBatchRequest) (*PredictBatchResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method PredictBatch not implemented")
}
func (UnimplementedModelServer) Finetune(context.Context, *FinetuneRequest) (*FinetuneResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Finetune not implemented")
}
func (UnimplementedModelServer) Save(context.Context, *SaveRequest) (*SaveResponse, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Save not implemented")
}

// UnsafeModelServer may be embedded to opt out of forward compatibility for this service.
// Use of this interface is not recommended, as added methods to ModelServer will
// result in compilation errors.
type UnsafeModelServer interface {
	mustEmbedUnimplementedModelServer()
}

func RegisterModelServer(s grpc.ServiceRegistrar, srv ModelServer) {
	s.RegisterService(&Model_ServiceDesc, srv)
}

func _Model_Predict_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PredictRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelServer).Predict(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Model_Predict_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelServer).Predict(ctx, req.(*PredictRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Model_PredictBatch_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PredictBatchRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelServer).PredictBatch(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Model_PredictBatch_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelServer).PredictBatch(ctx, req.(*PredictBatchRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Model_Finetune_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(FinetuneRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelServer).Finetune(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Model_Finetune_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelServer).Finetune(ctx, req.(*FinetuneRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _Model_Save_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(SaveRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ModelServer).Save(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: Model_Save_FullMethodName,
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ModelServer).Save(ctx, req.(*SaveRequest))
	}
	return interceptor(ctx, in, info, handler)
}

// Model_ServiceDesc is the grpc.ServiceDesc for Model service.
// It's only intended for direct use with grpc.RegisterService,
// and not to be introspected or modified (even as a copy)
var Model_ServiceDesc = grpc.ServiceDesc{
	ServiceName: "proto.Model",
	HandlerType: (*ModelServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "Predict",
			Handler:    _Model_Predict_Handler,
		},
		{
			MethodName: "PredictBatch",
			Handler:    _Model_PredictBatch_Handler,
		},
		{
			MethodName: "Finetune",
			Handler:    _Model_Finetune_Handler,
		},
		{
			MethodName: "Save",
			Handler:    _Model_Save_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "proto/model.proto",
}
