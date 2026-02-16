# 🚀 Chuleta de Comandos GPU

## 🟦 Compilación (NVCC)
Para compilar archivos de CUDA:
`nvcc nombre.cu -o salida`

## 🟩 Monitoreo de Hardware
| Objetivo | Comando |
| :--- | :--- |
| Uso básico | `nvidia-smi` |
| Actualización cada 1s | `nvidia-smi -l 1` |
| Ver procesos de memoria | `nvidia-smi --query-compute-apps=process_name,used_memory --format=csv` |

## 🟥 Profiling (Análisis de rendimiento)
`nsys profile ./programa`