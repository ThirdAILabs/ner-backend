package types

const contextLength = 20

type Entity struct {
	Label    string
	Text     string
	Start    int
	End      int
	LContext string
	RContext string
}

func (e *Entity) UpdateContext() {
	if e.LContext == "" {
		e.LContext = e.Text[max(0, e.Start-contextLength):e.Start]
	}
	if e.RContext == "" {
		e.RContext = e.Text[e.End:min(len(e.Text), e.End+contextLength)]
	}
}
