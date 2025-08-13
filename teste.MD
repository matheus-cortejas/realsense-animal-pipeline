# Projeto de Segmentação e Rastreamento 3D de Animais

## 1. Objetivo do Projeto

O objetivo deste projeto é processar arquivos de dados (`.bag`) de uma câmera Intel RealSense para **isolar, reconstruir em 3D e rastrear um animal específico** ao longo de um vídeo.

A pipeline automatiza a extração de dados, a segmentação 2D do animal usando YOLOv8, a reconstrução 3D da nuvem de pontos e o rastreamento (tracking) do animal no vídeo RGB.

---

## 2. Pipeline de Processamento

O projeto é dividido em três etapas principais:

### Etapa 1: Extração de Dados do Arquivo `.bag`

Esta etapa lê o arquivo `.bag` bruto e extrai os diferentes fluxos de dados, incluindo um vídeo RGB para visualização.

- **Script:** `src/bag_processor.py`
- **Entrada:** Um arquivo `.bag` gravado com uma câmera RealSense.
- **Processo:**
  1. Utiliza a biblioteca `pyrealsense2` para ler o arquivo `.bag`.
  2. Salva cada frame dos fluxos de **RGB**, **Profundidade (Depth)** e **Infravermelho (IR)**.
  3. Salva os parâmetros intrínsecos da câmera para alinhamento 3D.
  4. **Gera um vídeo `rgb_video.mp4`** a partir dos frames RGB extraídos.
- **Saída:** Um diretório estruturado (`data/generated/`) contendo:
  - `generatedRGB/`: Imagens de cor (`.png`)
  - `depth/`: Mapas de profundidade (`.npy`)
  - `generatedIR/`: Imagens de infravermelho (`.png`)
  - `camera_matrix_depth.npy` e `camera_matrix_color.npy`: Parâmetros da câmera.
  - **`rgb_video.mp4`**: Vídeo RGB para a etapa de tracking.

**Comando de Execução:**
```bash
python src/bag_processor.py --input data/raw_bags/NOME_DO_ARQUIVO.bag --output data/generated
```

### Etapa 2: Segmentação 2D e Reconstrução 3D

Esta etapa utiliza os dados extraídos para identificar o animal na imagem 2D e usar essa informação para "recortar" a nuvem de pontos 3D.

- **Script:** `src/segmenter.py`
- **Entrada:**
  - O diretório `data/generated/` criado na Etapa 1.
  - Um modelo de segmentação treinado (`.pt`), como o `yolov8n-seg.pt`.
- **Processo:**
  1. Carrega uma imagem (RGB ou IR) e seu mapa de profundidade correspondente.
  2. Usa o modelo **YOLOv8** para gerar uma **máscara de segmentação** 2D.
  3. Converte o mapa de profundidade em uma nuvem de pontos 3D.
  4. **Aplica a máscara 2D na nuvem de pontos 3D**, mantendo apenas os pontos do animal.
  5. Realiza um pós-processamento para limpar a nuvem de pontos.
- **Saída:** Um diretório de resultados (`results/`) contendo:
  - `masks/`: As máscaras de segmentação 2D (`.png`).
  - `pointclouds/`: As nuvens de pontos 3D do animal (`.pcd`).

**Comando de Execução:**
```bash
python src/segmenter.py --input_dir data/generated --output_dir results --model_path models/yolov8n-seg.pt
```

### Etapa 3: Rastreamento (Tracking) e Visualização

Esta etapa utiliza as máscaras geradas e o vídeo RGB para rastrear o animal ao longo do tempo, atribuindo um ID único a ele.

- **Script:** `src/track_and_visualize.py`
- **Entrada:**
  - O vídeo `rgb_video.mp4` (da Etapa 1).
  - O diretório `results/masks/` (da Etapa 2).
- **Processo:**
  1. Converte as máscaras de segmentação em *bounding boxes* para cada frame.
  2. Utiliza o algoritmo **SORT** para rastrear os objetos (animais) ao longo do vídeo.
  3. Desenha as *bounding boxes* e os IDs de rastreamento sobre o vídeo original.
- **Saída:**
  - `results/tracked_video.mp4`: Vídeo final com o rastreamento visualizado.
  - `results/failed_masks.txt`: Lista de frames onde a segmentação falhou.

**Comando de Execução:**
```bash
python src/track_and_visualize.py --video data/generated/rgb_video.mp4 --masks results/masks --output results/tracked_video.mp4
```

---

## 3. Como Executar a Pipeline Completa

1.  **Coloque seu arquivo `.bag`** no diretório `data/raw_bags/`.
2.  **Execute o processador de bag** para extrair os dados e o vídeo:
    ```bash
    python src/bag_processor.py --input data/raw_bags/SEU_ARQUIVO.bag --output data/generated
    ```
3.  **Execute o segmentador** para gerar as máscaras e as nuvens de pontos 3D:
    ```bash
    python src/segmenter.py --input_dir data/generated --output_dir results --model_path models/yolov8n-seg.pt
    ```
4.  **Execute o tracker** para gerar o vídeo com o rastreamento:
    ```bash
    python src/track_and_visualize.py --video data/generated/rgb_video.mp4 --masks results/masks --output results/tracked_video.mp4
    ```
5.  **Visualize os resultados**:
    - O vídeo `tracked_video.mp4` na pasta `results/`.
    - Os arquivos `.pcd` na pasta `results/pointclouds/` (com CloudCompare ou `teste2.py`).