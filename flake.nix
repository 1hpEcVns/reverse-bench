{
  description = "Fenwick tree (BIT) vs plain-array brute force crossover benchmark (C++23 / Rust edition 2024)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }: {
    devShells.x86_64-linux.default = let
      pkgs = nixpkgs.legacyPackages.x86_64-linux;
    in pkgs.mkShell {
      packages = with pkgs; [
        gcc
        rustc
        cargo
        python3
        python3Packages.matplotlib
        python3Packages.numpy
        python3Packages.pandas
      ];
    };
  };
}
