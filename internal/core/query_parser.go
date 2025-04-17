package core

import (
	"fmt"
	"math"

	"github.com/alecthomas/participle/v2"
)

/*
This is a parser for a simple query language with the following grammar:

Query       := Expr
Expr        := OrExpr ( "OR" OrExpr )*
OrExpr      := AndExpr ( "AND" AndExpr )*
AndExpr     := Condition | "NOT" Condition
Condition   := Filter | "(" Expr ")"
Filter 			:= Label Op Value
Label       := "COUNT" <identifier> | <identifier>
Op          := "CONTAINS" | "<" | ">" | "="
Value       := <string> | <int>

*/

var (
	parser = participle.MustBuild[QueryExpr](
		participle.Unquote("String"),
		participle.Union[Value](StringValue{}, IntValue{}),
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
	Label Label  ` @@`
	Op    string `@("CONTAINS" | "<" | ">" | "=" )`
	Value Value  `@@`
}

func (f *FilterExpr) ToFilter() (Filter, error) {
	if f.Label.Count {
		i, ok := f.Value.(IntValue)
		if !ok {
			return nil, fmt.Errorf("COUNT expr requires an int value to compare to")
		}

		switch f.Op {
		case "<":
			return &CountFilter{label: f.Label.Name, min: -1, max: i.Value}, nil
		case ">":
			return &CountFilter{label: f.Label.Name, min: i.Value, max: math.MaxInt}, nil
		case "=":
			return &CountFilter{label: f.Label.Name, min: i.Value - 1, max: i.Value + 1}, nil
		default:
			return nil, fmt.Errorf("invalid operator %s used with COUNT", f.Op)
		}
	}

	s, ok := f.Value.(StringValue)
	if !ok {
		return nil, fmt.Errorf("if not using COUNT operator then the value to compare to must be a string")
	}

	switch f.Op {
	case "CONTAINS":
		return &SubstringFilter{label: f.Label.Name, substr: s.Value}, nil
	case "<":
		return &StringLtFilter{label: f.Label.Name, value: s.Value}, nil
	case ">":
		return &StringGtFilter{label: f.Label.Name, value: s.Value}, nil
	case "=":
		return &StringEqFilter{label: f.Label.Name, value: s.Value}, nil
	default:
		return nil, fmt.Errorf("invalid operator %s used with string value", f.Op)
	}
}

func (f *FilterExpr) String() string {
	return fmt.Sprintf("%v %s %v", f.Label.String(), f.Op, f.Value)
}

type Label struct {
	Count bool   `@"COUNT"?`
	Name  string `@Ident` // TODO: Should this be string instead?
}

func (l *Label) String() string {
	if l.Count {
		return fmt.Sprintf("COUNT(%s)", l.Name)
	}
	return l.Name
}

type Value interface{ value() }

type StringValue struct {
	Value string `@String`
}

func (s StringValue) value() {}

type IntValue struct {
	Value int `@Int`
}

func (i IntValue) value() {}

// func main() {

// 	parser := participle.MustBuild[Query](
// 		participle.Unquote("String"),
// 		participle.Union[Value](StringValue{}, IntValue{}),
// 	)

// 	// q1 := `"SUBSTRING(x, ab)" AND	NOT "COUNT_GT(y, 2)" OR ("COUNT_LT(z, 5)" AND "SUBSTRING(z, 12)" OR "SUBSTRING(y, a94 9)")`

// 	q, err := parser.ParseString(
// 		"", `l1 CONTAINS "ab 2" AND	NOT COUNT l2 = 4 OR (l3 CONTAINS "abc" AND COUNT l4 < 10 OR l5 CONTAINS "9")`,
// 	)

// 	if err != nil {
// 		panic(err)
// 	}

// 	fmt.Printf("%+v\n", q)

// }
