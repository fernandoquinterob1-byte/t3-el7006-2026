# Third-party sources

`Neural-Process-Family/` is the upstream MIT-licensed PyTorch implementation
used by this project. It is intentionally kept as a source checkout because
the upstream repository is not packaged as a normal PyPI distribution.

To recreate it:

```powershell
git clone --depth 1 --filter=blob:none --sparse https://github.com/YannDubs/Neural-Process-Family.git Neural-Process-Family
git -C Neural-Process-Family sparse-checkout set npf
```

