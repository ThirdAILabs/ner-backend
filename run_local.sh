FILE_DIR=$(dirname "$0")

cd "$FILE_DIR/frontend"
npm i
npm run build

cd "$FILE_DIR"
go run cmd/local/main.go
