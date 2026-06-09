{
  description = "Agri Vision Edge development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            protobuf_25
            yaml-language-server
            nodejs  # marimo copilot

            uv

            # deployment tools
            usbsdmux
            bmaptool
            rsync

            # Needed by numpy/opencv wheels
            stdenv.cc.cc.lib
            zlib
          ];

          shellHook = ''
            export LD_LIBRARY_PATH=${
              pkgs.lib.makeLibraryPath [
                pkgs.stdenv.cc.cc.lib
                pkgs.zlib
                pkgs.libusb1
              ]
            }

            if [ ! -d .venv ]; then
                uv venv
            fi

            source .venv/bin/activate

            echo "Python: $(python --version)"

            echo "Agri Vision Edge dev shell"
          '';
        };
      });
}
