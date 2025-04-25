package utils_test

import (
	"fmt"
	"ner-backend/internal/core/utils"
	"testing"
	"time"
)

func TestRunInpool(t *testing.T) {
	worker := func(i int) (string, error) {
		if i%4 == 3 {
			time.Sleep(time.Duration(10-i) * time.Millisecond)
			return "", fmt.Errorf("error")
		}
		return fmt.Sprintf("%d-%d", i, i), nil
	}

	queue := make(chan int, 10)

	for i := 0; i < 10; i++ {
		queue <- i
	}

	close(queue)

	output := make(chan utils.CompletedTask[string], 10)

	utils.RunInPool(worker, queue, output, 5)

	success, errors := 0, 0
	for result := range output {
		if result.Error != nil {
			errors++
		} else {
			success++
		}
	}

	if success != 8 || errors != 2 {
		t.Fatal("invalid results")
	}
}
