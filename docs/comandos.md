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
    $raw = nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory,utilization.gpu,temperature.gpu --format=csv,noheader,nounits;
    $gpu = $raw.Split(',');
    
    Write-Host "--- MONITORIZACIÓN GPU EN TIEMPO REAL ---" -ForegroundColor Cyan;
    Write-Host "--------------------------------------------" -ForegroundColor Gray;
    Write-Host "| Core:  $($gpu[0].Trim().PadLeft(4)) MHz  | Mem: $($gpu[1].Trim().PadLeft(5)) MHz |" -ForegroundColor White;
    Write-Host "| Carga: $($gpu[2].Trim().PadLeft(4)) %    | Temp: $($gpu[3].Trim().PadLeft(5)) ºC  |" -ForegroundColor White;
    Write-Host "--------------------------------------------" -ForegroundColor Gray;
    Write-Host "[ Presiona Ctrl+C para detener ]" -ForegroundColor DarkGray;
    
    sleep -m 100;
}
```

## 🟥 Profiling (Análisis de rendimiento)
`nsys profile ./programa`