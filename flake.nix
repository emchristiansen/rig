{
  description = "Rig development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      rust-overlay
    }:
    flake-utils.lib.eachDefaultSystem (system:
      let 
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
      in
      { 
        devShells.default = with pkgs; mkShell {
          buildInputs = [
            pkg-config
            cmake
            just

            openssl
            sqlite
            postgresql
            protobuf

            rustToolchain
            wasm-bindgen-cli
            wasm-pack
          ];

          OPENSSL_DEV = openssl.dev;
          OPENSSL_LIB_DIR = "${openssl.out}/lib";
          OPENSSL_INCLUDE_DIR = "${openssl.dev}/include";

          shellHook = ''
            CANONICAL_WORKSPACE_ROOT="$(pwd -P; printf 'x')"
            CANONICAL_WORKSPACE_ROOT="''${CANONICAL_WORKSPACE_ROOT%x}"
            CANONICAL_WORKSPACE_ROOT="''${CANONICAL_WORKSPACE_ROOT%$'\n'}"
            WORKSPACE_PATH_HASH="$(printf '%s' "$CANONICAL_WORKSPACE_ROOT" | sha256sum)"
            WORKSPACE_PATH_HASH="''${WORKSPACE_PATH_HASH%% *}"
            CARGO_TARGET_CACHE="$HOME/.cache/cargo-target/rig-workspace-$WORKSPACE_PATH_HASH"

            unset CARGO_TARGET_DIR
            mkdir -p "$CARGO_TARGET_CACHE"

            if [ -L target ]; then
              CURRENT_TARGET="$(readlink target)"
              if [ "$CURRENT_TARGET" != "$CARGO_TARGET_CACHE" ]; then
                echo "Migrating target/ symlink: $CURRENT_TARGET → $CARGO_TARGET_CACHE"
                rm target
                ln -sfn "$CARGO_TARGET_CACHE" target
              fi
            elif [ -d target ]; then
              echo "ERROR: target/ is a real directory; refusing to replace or delete it." >&2
              echo "Resolve the preserved directory deliberately, then re-enter the workspace." >&2
              exit 1
            elif [ ! -e target ]; then
              ln -sfn "$CARGO_TARGET_CACHE" target
              echo "target/ → $CARGO_TARGET_CACHE (non-snapshotted cache dataset)"
            fi
          '';
        };
      }
    );
}
