# yet-essential

A collection of yet another essential and quality of life nodes for ComfyUI.

## Publishing to ComfyUI Registry

### Prerequisites

1. Set up a [Registry](https://registry.comfy.org) account
2. Create an API key at https://registry.comfy.org/nodes

### Steps to Publish

1. Install the comfy-cli tool:
   ```bash
   pip install comfy-cli
   ```

2. Verify your pyproject.toml has the correct metadata:
   ```toml
   [project]
   name = "your_extension_name"  # Use a unique name for your extension
   description = "Your extension description here."
   version = "0.1.0"  # Increment this with each update

   [tool.comfy]
   PublisherId = "your_publisher_id"  # Your Registry publisher ID
   DisplayName = "Your Extension Display Name"
   includes = ["dist/"]  # Include built React code (normally ignored by .gitignore)
   ```

3. Publish your extension:
   ```bash
   comfy-cli publish
   ```

4. When prompted, enter your API key

## License

Apache Software License 2.0
