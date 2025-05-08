FILE_DIR=$(dirname "$0")

cd "$FILE_DIR/frontend"
npm i --legacy-peer-deps
npm run build
rm -rf ../cmd/local/out
mv out/ ../cmd/local/out/

cd ..
go run cmd/local/main.go
