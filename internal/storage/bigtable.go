package storage

import (
	"bytes"
	"context"
	"io"
	"strings"

	"cloud.google.com/go/bigtable"
)

type BigTableProvider struct {
	client *bigtable.Client
	tableName string
	columnFamilyName string
	columnName string
}

type BigTableProviderConfig struct {
	Project string
	Instance string
	TableName string
	ColumnFamilyName string
	ColumnName string
}

const BigTablePrefix = "bigtable://"

func NewBigTableProvider(cfg BigTableProviderConfig) (*BigTableProvider, error) {
	client, err := bigtable.NewClient(context.Background(), cfg.Project, cfg.Instance)
	if err != nil {
		return nil, err
	}

	return &BigTableProvider{client: client, tableName: cfg.TableName, columnFamilyName: cfg.ColumnFamilyName, columnName: cfg.ColumnName}, nil
}

func (p *BigTableProvider) IterObjects(ctx context.Context, bucket, prefix string) ObjectIterator {
	table := p.client.Open(p.tableName)

	iterator := func(yield func(obj Object, err error) bool) {
		table.ReadRows(ctx, bigtable.PrefixRange(p.columnName), func(row bigtable.Row) bool {
			item := row[p.columnFamilyName][0]
			return yield(Object{Name: BigTablePrefix + item.Row, Size: int64(len(item.Value))}, nil)
		}, bigtable.RowFilter(bigtable.ColumnFilter(p.columnName)))
	}
	return iterator
}

func (p *BigTableProvider) GetObjectStream(bucket, key string) (io.Reader, error) {
	table := p.client.Open(p.tableName)
	row, err := table.ReadRow(context.Background(), strings.TrimPrefix(key, BigTablePrefix))
	if err != nil {
		return nil, err
	}
	return bytes.NewReader(row[p.columnFamilyName][0].Value), nil
}

// The following are not needed for inference. Therefore we will not implement them.

func (p *BigTableProvider) CreateBucket(ctx context.Context, bucket string) error {
	return nil
}

func (p *BigTableProvider) GetObject(ctx context.Context, bucket, key string) ([]byte, error) {
	return nil, nil
}

func (p *BigTableProvider) PutObject(ctx context.Context, bucket, key string, data io.Reader) error {
	return nil
}

func (p *BigTableProvider) DownloadDir(ctx context.Context, bucket, prefix, dest string, overwrite bool) error {
	return nil
}

func (p *BigTableProvider) UploadDir(ctx context.Context, bucket, prefix, src string) error {
	return nil
}

func (p *BigTableProvider) ListObjects(ctx context.Context, bucket, prefix string) ([]Object, error) {
	return nil, nil
}

func (p *BigTableProvider) ValidateAccess(ctx context.Context, bucket, prefix string) error {
	return nil
}