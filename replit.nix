{ pkgs }: {
  deps = [
    pkgs.python311Full   # Python 3.11 + pip
    pkgs.cacert
    pkgs.glibcLocales
  ];
}
