PROTO_DIR := protos
PROTO_OUT := src/schema
PROTOS    := $(wildcard $(PROTO_DIR)/*.proto)

.PHONY: dev test docs protos clean-protos

# dev environment: venv + deps (incl. novaad in editable mode) + generated protobuf code
dev:
	@command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
	@command -v protoc >/dev/null || { echo "protoc not found: brew install protobuf"; exit 1; }
	uv sync --all-groups
	$(MAKE) protos

# core.py holds the test_* functions; pytest collects explicitly listed files regardless of name
test:
	uv run pytest test src/novaad/core.py

# live-reloading docs server on http://localhost:8000
docs:
	uv run --group docs zensical serve

protos: $(PROTOS)
	mkdir -p $(PROTO_OUT)
	protoc --proto_path=$(PROTO_DIR) --python_out=$(PROTO_OUT) --pyi_out=$(PROTO_OUT) $(PROTOS)

clean-protos:
	rm -rf $(PROTO_OUT)
