package utils

import "sync"

type CompletedTask[T any] struct {
	Result T
	Error  error
}

func RunInPool[In any, Out any](worker func(In) (Out, error), queue chan In, completed chan CompletedTask[Out], maxWorkers int) {
	workers := min(len(queue), maxWorkers)

	go func() {
		wg := sync.WaitGroup{}
		wg.Add(workers)

		for i := 0; i < workers; i++ {
			go func() {
				defer wg.Done()

				for {
					next, ok := <-queue
					if !ok {
						return
					}

					res, err := worker(next)
					if err != nil {
						completed <- CompletedTask[Out]{Error: err}
					} else {
						completed <- CompletedTask[Out]{Result: res, Error: nil}
					}
				}
			}()
		}

		wg.Wait()

		close(completed)
	}()
}
