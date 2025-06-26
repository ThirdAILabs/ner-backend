package api

import (
	"ner-backend/internal/core"
	"ner-backend/internal/core/python"
	"ner-backend/internal/database"
	"ner-backend/pkg/api"
)

func convertModel(m database.Model) api.Model {
	model := api.Model{
		Id:           m.Id,
		Name:         m.Name,
		Status:       m.Status,
		CreationTime: m.CreationTime,
	}
	if m.BaseModelId.Valid {
		model.BaseModelId = &m.BaseModelId.UUID
	}

	for _, tag := range m.Tags {
		model.Tags = append(model.Tags, tag.Tag)
	}

	switch core.ParseModelType(m.Type) {
	case core.Presidio:
		model.Finetunable = false
	case core.BoltUdt:
		model.Finetunable = true
	case core.PythonTransformer, core.PythonCnn, core.OnnxCnn:
		model.Finetunable = python.PythonPluginEnabled()
	}
	return model
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
		Id:                 r.Id,
		Model:              convertModel(*r.Model),
		ReportName:         r.ReportName,
		StorageType:        r.StorageType,
		StorageParams:      r.StorageParams,
		IsUpload:           r.IsUpload,
		Stopped:            r.Stopped,
		CreationTime:       r.CreationTime,
		Groups:             convertGroups(r.Groups),
		TotalFileCount:     int(r.TotalFileCount),
		SucceededFileCount: int(r.SucceededFileCount),
		FailedFileCount:    int(r.FailedFileCount),
	}
	report.TagCounts = make(map[string]uint64)

	for _, tag := range r.Tags {
		report.Tags = append(report.Tags, tag.Tag)
		report.TagCounts[tag.Tag] = tag.Count
	}

	if r.CustomTags != nil {
		report.CustomTags = make(map[string]string)
		for _, tag := range r.CustomTags {
			report.CustomTags[tag.Tag] = tag.Pattern
			report.TagCounts[tag.Tag] = tag.Count
		}
	}

	if r.ShardDataTask != nil {
		report.ShardDataTaskStatus = r.ShardDataTask.Status
	}

	for _, err := range r.Errors {
		report.Errors = append(report.Errors, err.Error)
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
		Object:   e.Object,
		Start:    e.Start,
		End:      e.End,
		Label:    e.Label,
		Text:     e.Text,
		LContext: e.LContext,
		RContext: e.RContext,
	}
}

func convertEntities(es []database.ObjectEntity) []api.Entity {
	var entities []api.Entity
	for _, e := range es {
		entities = append(entities, convertEntity(e))
	}
	return entities
}
