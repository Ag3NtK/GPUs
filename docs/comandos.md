# 🚀 Chuleta de Comandos GPU

## 🟦 Compilación (NVCC)
Para compilar archivos de CUDA:
`nvcc nombre.cu -o salida`

## 🟩 Monitoreo de Hardware
| Objetivo | Comando |
| :--- | :--- |
| Uso básico | `nvidia-smi` |
| Uso detallado | `nvidia-smi -q` |
| Actualización cada 1s | `nvidia-smi -l 1` |

## 📊 Monitor de Rendimiento
Para ver los relojes y la carga de forma limpia en Windows PowerShell:
```powershell
while($true) { 
    cls; 
    $gpu = (nvidia-smi --query-gpu=clocks.gr,clocks.mem,utilization.gpu,temp --format=csv,noheader,nounits).Split(',');
    Write-Host "ESTADO DE LA GPU" -ForegroundColor Cyan;
    Write-Host "--------------------------------------------";
    Write-Host "| Core: $($gpu[0]) MHz | Mem: $($gpu[1]) MHz |";
    Write-Host "| Carga: $($gpu[2]) %   | Temp: $($gpu[3]) ºC |";
    Write-Host "--------------------------------------------";
    sleep -m 500;
}
```

## 🟥 Profiling (Análisis de rendimiento)
`nsys profile ./programa`