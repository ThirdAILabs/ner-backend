package core

import (
	"fmt"
	"math"

	"github.com/alecthomas/participle/v2"
	"gorm.io/gorm"
)

/*
This is a parser for a simple query language with the following grammar:

Query       := Expr
Expr        := OrExpr ( "OR" OrExpr )*
OrExpr      := AndExpr ( "AND" AndExpr )*
AndExpr     := Condition | "NOT" Condition
Condition   := Filter | "(" Expr ")"
Filter 			:= "COUNT(" <identifier> ")" IntOp <int> | <identifier> StrOp <string>
IntOp       := "<" | ">" | "="
StrOp       := "CONTAINS" | "<" | ">" | "="

*/

var (
	parser = participle.MustBuild[QueryExpr](
		participle.Unquote("String"),
		// participle.Union[Value](StringValue{}, IntValue{}),
	)
)

func ParseQuery(query string) (Filter, error) {
	q, err := parser.ParseString("", query)
	if err != nil {
		return nil, fmt.Errorf("error parsing query '%s': %w", query, err)
	}

	filter, err := q.ToFilter()
	if err != nil {
		return nil, fmt.Errorf("error converting query '%s' to filter: %w", query, err)
	}

	return filter, nil
}

func ToSql(db *gorm.DB, query string) (*gorm.DB, error) {
	q, err := parser.ParseString("", query)
	if err != nil {
		return nil, fmt.Errorf("error parsing query '%s': %w", query, err)
	}

	expr, err := q.Expr.ToSql(db)
	if err != nil {
		return nil, fmt.Errorf("error converting query '%s' to SQL: %w", query, err)
	}

	return expr, nil
}

type QueryExpr struct {
	Expr *Expr `@@`
}

func (q *QueryExpr) ToFilter() (Filter, error) {
	return q.Expr.ToFilter()
}

func (q *QueryExpr) String() string {
	return q.Expr.String()
}

type Expr struct {
	Ors []*OrExpr `@@ ( "OR" @@ )*`
}

func (q *Expr) ToFilter() (Filter, error) {
	if len(q.Ors) == 0 {
		return nil, fmt.Errorf("empty OR expression")
	}

	if len(q.Ors) == 1 {
		return q.Ors[0].ToFilter()
	}

	var filters []Filter
	for _, cond := range q.Ors {
		f, err := cond.ToFilter()
		if err != nil {
			return nil, err
		}
		filters = append(filters, f)
	}

	return &OrFilter{filters: filters}, nil
}

func (e *Expr) ToSql(db *gorm.DB) (*gorm.DB, error) {
	expr := db
	for _, cond := range e.Ors {
		subexpr, err := cond.ToSql(db)
		if err != nil {
			return nil, err
		}
		expr = expr.Or(subexpr)
	}

	return expr, nil
}

func (e *Expr) String() string {
	if len(e.Ors) == 0 {
		return ""
	}

	if len(e.Ors) == 1 {
		return e.Ors[0].String()
	}

	out := fmt.Sprintf("(%s)", e.Ors[0].String())
	for _, cond := range e.Ors[1:] {
		out += fmt.Sprintf(" OR (%s)", cond.String())
	}

	return out
}

type OrExpr struct {
	Ands []*Condition `@@ ( "AND" @@ )*`
}

func (o *OrExpr) ToFilter() (Filter, error) {
	if len(o.Ands) == 0 {
		return nil, fmt.Errorf("empty AND expression")
	}

	if len(o.Ands) == 1 {
		return o.Ands[0].ToFilter()
	}

	var filters []Filter
	for _, cond := range o.Ands {
		f, err := cond.ToFilter()
		if err != nil {
			return nil, err
		}
		filters = append(filters, f)
	}

	return &AndFilter{filters: filters}, nil
}

func (o *OrExpr) ToSql(db *gorm.DB) (*gorm.DB, error) {
	expr := db
	for _, cond := range o.Ands {
		subexpr, err := cond.ToSql(db)
		if err != nil {
			return nil, err
		}
		expr = expr.Where(subexpr)
	}

	return expr, nil
}

func (e *OrExpr) String() string {
	if len(e.Ands) == 0 {
		return ""
	}

	if len(e.Ands) == 1 {
		return e.Ands[0].String()
	}

	out := fmt.Sprintf("(%s)", e.Ands[0].String())
	for _, cond := range e.Ands[1:] {
		out += fmt.Sprintf(" AND (%s)", cond.String())
	}

	return out
}

type Condition struct {
	Not     bool        `@"NOT"?`
	Filter  *FilterExpr ` @@`
	SubExpr *Expr       `| "(" @@ ")" `
}

func (c *Condition) ToFilter() (Filter, error) {
	var filter Filter = nil
	var err error
	if c.Filter != nil {
		filter, err = c.Filter.ToFilter()
	} else if c.SubExpr != nil {
		filter, err = c.SubExpr.ToFilter()
	}

	if err != nil {
		return nil, err
	}

	if c.Not {
		filter = &NotFilter{filter: filter}
	}

	return filter, nil
}

func (c *Condition) ToSql(db *gorm.DB) (*gorm.DB, error) {
	var expr *gorm.DB
	var err error

	if c.Filter != nil {
		expr, err = c.Filter.ToSql(db)
	} else {
		expr, err = c.SubExpr.ToSql(db)
	}

	if err != nil {
		return nil, err
	}

	if c.Not {
		return db.Not(expr), nil
	}
	return expr, nil
}

func (c *Condition) String() string {
	var out string
	if c.SubExpr != nil {
		out = c.SubExpr.String()
	} else {
		out = c.Filter.String()
	}
	if c.Not {
		return fmt.Sprintf("NOT (%s)", out)
	}
	return out
}

type FilterExpr struct {
	CountFilter  *CountFilterExpr  `"COUNT" @@`
	StringFilter *StringFilterExpr `| @@`
}

func (f *FilterExpr) ToFilter() (Filter, error) {
	if f.CountFilter != nil {
		return f.CountFilter.ToFilter()
	} else if f.StringFilter != nil {
		return f.StringFilter.ToFilter()
	} else {
		return nil, fmt.Errorf("invalid filter expression")
	}
}

func (f *FilterExpr) ToSql(db *gorm.DB) (*gorm.DB, error) {
	if f.CountFilter != nil {
		return f.CountFilter.ToSql(db)
	} else if f.StringFilter != nil {
		return f.StringFilter.ToSql(db)
	} else {
		return nil, fmt.Errorf("invalid filter expression")
	}
}

func (f *FilterExpr) String() string {
	if f.CountFilter != nil {
		return f.CountFilter.String()
	} else if f.StringFilter != nil {
		return f.StringFilter.String()
	} else {
		return "<invalid filter expression>"
	}
}

type CountFilterExpr struct {
	Label string `"(" @Ident")"`
	Op    string `@("<" | ">" | "=" )`
	Value int    `@Int`
}

func (c *CountFilterExpr) ToFilter() (Filter, error) {
	switch c.Op {
	case "<":
		return &CountFilter{label: c.Label, min: -1, max: c.Value}, nil
	case ">":
		return &CountFilter{label: c.Label, min: c.Value, max: math.MaxInt}, nil
	case "=":
		return &CountFilter{label: c.Label, min: c.Value - 1, max: c.Value + 1}, nil
	default:
		return nil, fmt.Errorf("invalid operator %s used with COUNT", c.Op)
	}
}

func (c *CountFilterExpr) ToSql(db *gorm.DB) (*gorm.DB, error) {
	subquery := db.Table("object_entities as o").Select("COUNT(*)").Where("o.object = object AND o.label = ?", c.Label)

	switch c.Op {
	case "<":
		return db.Where("(?) < ?", subquery, c.Value), nil
	case ">":
		return db.Where("(?) > ?", subquery, c.Value), nil
	case "=":
		return db.Where("(?) = ?", subquery, c.Value), nil
	default:
		return nil, fmt.Errorf("invalid operator %s used with string value", c.Op)
	}
}

func (c *CountFilterExpr) String() string {
	return fmt.Sprintf("COUNT(%s) %s %d", c.Label, c.Op, c.Value)
}

type StringFilterExpr struct {
	Label string `@Ident`
	Op    string `@("CONTAINS" | "<" | ">" | "=" )`
	Value string `@String`
}

func (s *StringFilterExpr) ToFilter() (Filter, error) {
	switch s.Op {
	case "CONTAINS":
		return &SubstringFilter{label: s.Label, substr: s.Value}, nil
	case "<":
		return &StringLtFilter{label: s.Label, value: s.Value}, nil
	case ">":
		return &StringGtFilter{label: s.Label, value: s.Value}, nil
	case "=":
		return &StringEqFilter{label: s.Label, value: s.Value}, nil
	default:
		return nil, fmt.Errorf("invalid operator %s used with string value", s.Op)
	}
}

func (s *StringFilterExpr) ToSql(db *gorm.DB) (*gorm.DB, error) {
	switch s.Op {
	case "CONTAINS":
		return db.Where("label = ? AND text LIKE ?", s.Label, "%"+s.Value+"%"), nil
	case "<":
		return db.Where("label = ? AND text < ?", s.Label, s.Value), nil
	case ">":
		return db.Where("label = ? AND text > ?", s.Label, s.Value), nil
	case "=":
		return db.Where("label = ? AND text = ?", s.Label, s.Value), nil
	default:
		return nil, fmt.Errorf("invalid operator %s used with string value", s.Op)
	}
}

func (s *StringFilterExpr) String() string {
	return fmt.Sprintf("%s %s %s", s.Label, s.Op, s.Value)
}
