# Dashboard Streamlit para o contador de veículos (deteccao_carros.py)
#
# Como rodar (na mesma pasta do deteccao_carros.py):
#   pip install streamlit pandas
#   streamlit run app_streamlit.py

import cv2
import pandas as pd
import streamlit as st

import deteccao_carros as dc  # reaproveita configs e funções auxiliares

# Nomes amigáveis para as classes COCO consideradas "veículo"
CLASS_NAMES = {2: "Carro", 3: "Moto", 5: "Ônibus", 7: "Caminhão"}

st.set_page_config(page_title="Contagem de Veículos", page_icon="🚗", layout="wide")


# ----------------------------- Detecção ----------------------------- #

def run_detection(frame_slot, metric_slot, progress_slot, display_every: int) -> None:
    """
    Roda o YOLO + tracking frame a frame, atualiza a tela ao vivo e
    guarda cada veículo contado em st.session_state["events"].
    """
    model = dc.load_model(dc.MODEL_PATH)  # modelo novo = tracker zerado
    cap = dc.load_video(dc.VIDEO_PATH)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    line_y = int(height * dc.LINE_Y_RATIO)

    st.session_state["fps"] = fps
    events: list = []
    st.session_state["events"] = events  # mesma lista: atualiza em tempo real

    counted_ids = set()
    last_centers = {}
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=dc.CONFIDENCE,
                verbose=False,
            )

            detections = dc.extract_detections(results[0]) if results else None
            if detections is not None:
                boxes, ids, classes, confs = detections
                for box, obj_id, cls, conf in zip(boxes, ids, classes, confs):
                    if cls not in dc.VEHICLE_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    prev_center = last_centers.get(obj_id)
                    if dc.cross_line(prev_center, center[1], line_y) and obj_id not in counted_ids:
                        counted_ids.add(obj_id)
                        events.append(
                            {
                                "ID": obj_id,
                                "Classe": CLASS_NAMES.get(cls, str(cls)),
                                "Frame": frame_idx,
                                "Tempo (s)": round(frame_idx / fps, 2),
                            }
                        )

                    last_centers[obj_id] = center
                    dc.draw_annotations(frame, box, center, obj_id, cls, conf)

            dc.draw_overlay(frame, width, line_y, len(events))

            # Atualiza a tela só a cada N frames (deixa o processamento mais rápido)
            if frame_idx % display_every == 0:
                frame_slot.image(frame, channels="BGR", width="stretch")
                metric_slot.metric("Veículos contados", len(events))
                if total_frames > 0:
                    progress_slot.progress(min(frame_idx / total_frames, 1.0))
    finally:
        cap.release()

    metric_slot.metric("Veículos contados", len(events))
    progress_slot.progress(1.0)


# ----------------------------- Relatório ----------------------------- #

def render_report() -> None:
    events = st.session_state.get("events")
    if events is None:
        st.info("Clique em **Iniciar** para processar o vídeo.")
        return

    st.divider()
    st.header("📊 Relatório final")

    if not events:
        st.warning("Nenhum veículo cruzou a linha de contagem.")
        return

    df = pd.DataFrame(events)
    counts = df["Classe"].value_counts().reindex(list(CLASS_NAMES.values()), fill_value=0)

    # Métricas: total + uma por tipo de veículo
    cols = st.columns(len(CLASS_NAMES) + 1)
    cols[0].metric("Total", len(df))
    for col, (name, value) in zip(cols[1:], counts.items()):
        col.metric(name, int(value))

    left, right = st.columns(2)

    with left:
        st.subheader("Por tipo de veículo")
        st.bar_chart(counts)

    with right:
        st.subheader("Contagem acumulada ao longo do tempo")
        timeline = df.sort_values("Tempo (s)").reset_index(drop=True)
        timeline["Acumulado"] = timeline.index + 1
        st.line_chart(timeline, x="Tempo (s)", y="Acumulado")

    st.subheader("Veículos contados")
    st.dataframe(df, hide_index=True)
    st.download_button(
        "⬇️ Baixar CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="veiculos_contados.csv",
        mime="text/csv",
    )


# ----------------------------- Interface ----------------------------- #

st.title("🚗 Contagem de veículos com YOLO")

with st.sidebar:
    st.header("Configurações")
    display_every = st.slider(
        "Atualizar imagem a cada N frames",
        min_value=1,
        max_value=10,
        value=2,
        help="Valores maiores deixam o processamento mais rápido.",
    )
    st.caption(f"**Vídeo:** {dc.VIDEO_PATH.name}")
    st.caption(f"**Modelo:** {dc.MODEL_PATH}")
    st.caption(f"**Confiança mínima:** {dc.CONFIDENCE}")
    st.caption(f"**Direção:** {dc.DIRECAO_CONTAGEM}")

btn_start, btn_stop, _ = st.columns([1, 1, 6])
start = btn_start.button("▶️ Iniciar", type="primary")
btn_stop.button("⏹️ Parar")  # clicar reinicia o script e interrompe o loop

frame_slot = st.empty()
metric_slot = st.empty()
progress_slot = st.empty()

if start:
    try:
        run_detection(frame_slot, metric_slot, progress_slot, display_every)
        st.success("Processamento concluído!")
    except (FileNotFoundError, RuntimeError) as exc:
        st.error(f"Erro: {exc}")

render_report()