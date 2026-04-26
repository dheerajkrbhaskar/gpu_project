# Sample Images

The current sample dataset lives under `D/`.

Useful subfolders:

- `D/UD/` for mostly clean or undamaged road images
- `D/CD/` for crack-damaged road images

Recommended demo images:

- `D/UD/7001-12.jpg` for a clean baseline example
- `D/CD/7001-41.jpg` for a crack example
- `D/UD/7036-51.jpg` and `D/CD/7036-118.jpg` for extra comparison runs

Example usage:

```bash
./build/gpu_crack_detection --input data/D/UD/7001-12.jpg --mode gpu --repeat 5
```

All generated results are written into `outputs/` as PNG files, along with the timing text printed in the terminal.
