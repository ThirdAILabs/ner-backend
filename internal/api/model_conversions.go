package api

import (
	"ner-backend/internal/database"
	"ner-backend/pkg/api"
)

func convertModel(m database.Model) api.Model {
	return api.Model{
		Id:     m.Id,
		Name:   m.Name,
		Type:   m.Type,
		Status: m.Status,
	}
}

func convertModels(ms []database.Model) []api.Model {
	models := make([]api.Model, 0, len(ms))
	for _, m := range ms {
		models = append(models, convertModel(m))
	}
	return models
}

func convertGroup(g database.Group) api.Group {
	var objects []string
	for _, obj := range g.Objects {
		objects = append(objects, obj.Object)
	}
	return api.Group{
		Id:      g.Id,
		Name:    g.Name,
		Query:   g.Query,
		Objects: objects,
	}
}

func convertGroups(gs []database.Group) []api.Group {
	var groups []api.Group
	for _, g := range gs {
		groups = append(groups, convertGroup(g))
	}
	return groups
}

func convertReport(r database.Report) api.Report {
	report := api.Report{
		Id:             r.Id,
		Model:          convertModel(*r.Model),
		SourceS3Bucket: r.SourceS3Bucket,
		SourceS3Prefix: r.SourceS3Prefix.String,
		CreationTime:   r.CreationTime,
		Groups:         convertGroups(r.Groups),
	}

	if r.ShardDataTask != nil {
		report.ShardDataTaskStatus = r.ShardDataTask.Status
	}

	return report
}

func convertReports(rs []database.Report) []api.Report {
	reports := make([]api.Report, 0, len(rs))
	for _, r := range rs {
		reports = append(reports, convertReport(r))
	}
	return reports
}

func convertEntity(e database.ObjectEntity) api.Entity {
	return api.Entity{
		Object: e.Object,
		Start:  e.Start,
		End:    e.End,
		Label:  e.Label,
		Text:   e.Text,
	}
}

func convertEntities(es []database.ObjectEntity) []api.Entity {
	var entities []api.Entity
	for _, e := range es {
		entities = append(entities, convertEntity(e))
	}
	return entities
}
