import streamlit as st
import numpy as np
import os
import glob
import requests
import zipfile
from pathlib import Path
import io
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(
    page_title="EcoClean – Option 1 | AI Graded",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(160deg,#0d1b2a 0%,#1b2838 60%,#0d2137 100%); color:#e0e1dd; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0d1b2a,#122030) !important; border-right:1px solid rgba(111,255,233,0.12); }
[data-testid="stSidebar"] * { color:#c8d6e5; }
.hero-title { font-size:2.8rem;font-weight:800;letter-spacing:-1px;background:linear-gradient(90deg,#6fffe9,#5bc0be,#3a506b);-webkit-background-clip:text;-webkit-text-fill-color:transparent; }
.hero-sub { color:rgba(111,255,233,0.7);font-size:1.1rem;margin-top:4px;margin-bottom:1.5rem; }
.divider { height:2px;background:linear-gradient(90deg,transparent,rgba(91,192,190,0.4),transparent);margin:24px 0; }
.card { background:rgba(255,255,255,0.04);border:1px solid rgba(111,255,233,0.15);border-radius:14px;padding:20px 24px;margin-bottom:14px;box-shadow:0 4px 20px rgba(0,0,0,0.3); }
.card-label { font-size:0.78rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#5bc0be;margin-bottom:4px; }
.card-val { font-size:2.2rem;font-weight:800;color:#6fffe9;margin:0; }
.task-hdr { background:rgba(91,192,190,0.09);border-left:4px solid #5bc0be;border-radius:8px;padding:12px 18px;margin-bottom:14px; }
.task-num { color:#6fffe9;font-weight:700;font-size:0.88rem;text-transform:uppercase;letter-spacing:1px; }
div.stButton>button { background:linear-gradient(90deg,#5bc0be,#3a506b)!important;color:#fff!important;border:none!important;border-radius:25px!important;font-weight:600!important;padding:.55rem 1.8rem!important;box-shadow:0 4px 12px rgba(91,192,190,0.35)!important;transition:all .25s ease!important; }
div.stButton>button:hover { transform:translateY(-2px)!important;box-shadow:0 8px 20px rgba(91,192,190,0.55)!important; }
.badge-r { display:inline-block;padding:9px 24px;border-radius:50px;background:linear-gradient(90deg,#2ecc71,#27ae60);color:#fff;font-weight:700;font-size:1.1rem;box-shadow:0 4px 14px rgba(46,204,113,0.4); }
.badge-o { display:inline-block;padding:9px 24px;border-radius:50px;background:linear-gradient(90deg,#e67e22,#d35400);color:#fff;font-weight:700;font-size:1.1rem;box-shadow:0 4px 14px rgba(230,126,34,0.4); }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "o-vs-r-split")
TRAIN_PATH = os.path.join(DATA_DIR, "train")
TEST_PATH  = os.path.join(DATA_DIR, "test")
MODEL1     = os.path.join(BASE_DIR, "O_R_tlearn_vgg16.keras")
MODEL2     = os.path.join(BASE_DIR, "O_R_tlearn_fine_tune_vgg16.keras")
IMG_SZ, BS, SEED, VS = 150, 32, 42, 0.2

# ── Lazy TF ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_tf():
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras import optimizers
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import vgg16
    from sklearn import metrics as skm
    return tf, Sequential, Model, optimizers, EarlyStopping, ModelCheckpoint, LearningRateScheduler, Dense, Dropout, ImageDataGenerator, vgg16, skm

# ── Helpers ──────────────────────────────────────────────────────────────────
def download_data():
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/kd6057VPpABQ2FqCbgu9YQ/o-vs-r-split-reduced-1200.zip"
    zp  = os.path.join(BASE_DIR, "data.zip")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zp,"wb") as f:
            for chunk in r.iter_content(8192): f.write(chunk)
    with zipfile.ZipFile(zp) as z: z.extractall(BASE_DIR)
    os.remove(zp)

@st.cache_resource(show_spinner=False)
def get_generators():
    tf,_,_,_,_,_,_,_,_,IDG,_,_ = get_tf()
    tr  = IDG(validation_split=VS,rescale=1/255.,width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)
    vl  = IDG(validation_split=VS,rescale=1/255.)
    ts  = IDG(rescale=1/255.)
    tg  = tr.flow_from_directory(TRAIN_PATH,seed=SEED,batch_size=BS,class_mode="binary",shuffle=True,target_size=(IMG_SZ,IMG_SZ),subset="training")
    vg  = vl.flow_from_directory(TRAIN_PATH,seed=SEED,batch_size=BS,class_mode="binary",shuffle=True,target_size=(IMG_SZ,IMG_SZ),subset="validation")
    tsg = ts.flow_from_directory(TEST_PATH, seed=SEED,batch_size=BS,class_mode="binary",shuffle=False,target_size=(IMG_SZ,IMG_SZ))
    return tg,vg,tsg

def build_model(fine_tune=False):
    tf,Sequential,Model,_,_,_,_,Dense,Dropout,_,vgg16,_ = get_tf()
    vgg = vgg16.VGG16(include_top=False,weights="imagenet",input_shape=(IMG_SZ,IMG_SZ,3))
    out = tf.keras.layers.Flatten()(vgg.layers[-1].output)
    bm  = Model(vgg.input, out)
    for l in bm.layers: l.trainable = False
    if fine_tune:
        s=False
        for l in bm.layers:
            if l.name=="block5_conv3": s=True
            l.trainable=s
    m = Sequential([bm,Dense(512,activation="relu"),Dropout(0.3),Dense(512,activation="relu"),Dropout(0.3),Dense(1,activation="sigmoid")])
    return m, bm

@st.cache_resource(show_spinner=False)
def load_models():
    tf,*_ = get_tf()
    return tf.keras.models.load_model(MODEL1), tf.keras.models.load_model(MODEL2)

def exp_decay(e): return 1e-4*np.exp(-0.1*e)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ♻️ EcoClean AI")
st.sidebar.markdown("**Option 1 – AI Graded Pipeline**")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation",[
    "🏠 Overview",
    "⚙️ Tasks 1–5: Setup & Model",
    "📈 Tasks 6–8: Training Curves",
    "🖼️ Tasks 9–10: Predictions",
])

# ══════════════════ OVERVIEW ══════════════════════════════════════════════════
if page=="🏠 Overview":
    st.markdown('<p class="hero-title">EcoClean AI · Waste Classifier</p>',unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Option 1 – AI Graded Submission · VGG16 Transfer Learning & Fine-Tuning</p>',unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    for col,lbl,val,sub in [(c1,"Architecture","VGG16","ImageNet Pre-trained"),(c2,"Classes","2","Organic · Recyclable"),(c3,"Tasks","10","All Required")]:
        col.markdown(f'<div class="card"><div class="card-label">{lbl}</div><div class="card-val">{val}</div><small style="color:#8d99ae">{sub}</small></div>',unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown("### 📋 Task Checklist")
    tasks=[("Task 1","Print TensorFlow version"),("Task 2","Create test_generator"),("Task 3","Print length of train_generator"),("Task 4","Print model summary"),("Task 5","Compile the model"),("Task 6","Accuracy curves – Extract Features"),("Task 7","Loss curves – Fine-Tune"),("Task 8","Accuracy curves – Fine-Tune"),("Task 9","Plot test image – Extract Features (index=1)"),("Task 10","Plot test image – Fine-Tuned (index=1)")]
    l,r = st.columns(2)
    for i,(n,d) in enumerate(tasks):
        (l if i%2==0 else r).markdown(f'<div class="task-hdr"><span class="task-num">✅ {n}</span><br><span style="color:#c8d6e5">{d}</span></div>',unsafe_allow_html=True)
    st.info("👉 Start with **Tasks 1–5** to download data & train the model, then proceed through the sidebar.")

# ══════════════════ TASKS 1-5 ═════════════════════════════════════════════════
elif page=="⚙️ Tasks 1–5: Setup & Model":
    st.markdown("## ⚙️ Tasks 1–5: Environment Setup & Model Build")
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)

    # Task 1 — always visible
    st.markdown('<div class="task-hdr"><span class="task-num">✅ Task 1 · TensorFlow Version</span></div>',unsafe_allow_html=True)
    tf,*_ = get_tf()
    st.success(f"**TensorFlow Version: {tf.__version__}**")
    st.code(f"import tensorflow as tf\nprint(tf.__version__)  # → {tf.__version__}",language="python")
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)

    # Auto-download dataset
    st.markdown("### 📥 Downloading Dataset")
    if not os.path.exists(DATA_DIR):
        with st.spinner("⏳ Downloading waste classification dataset from IBM Skills Network (this happens only once)…"):
            try:
                download_data()
                st.success("✅ Dataset downloaded and extracted!")
            except Exception as e:
                st.error(f"Download failed: {e}")
                st.stop()
    else:
        st.success("✅ Dataset already on disk — ready to use!")

    # Build generators — auto
    with st.spinner("Building data generators…"):
        train_gen, val_gen, test_gen = get_generators()

    # Task 2
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown('<div class="task-hdr"><span class="task-num">✅ Task 2 · test_generator</span></div>',unsafe_allow_html=True)
    st.code("""test_generator = test_datagen.flow_from_directory(
    directory=path_test, class_mode='binary', seed=seed,
    batch_size=batch_size, shuffle=False, target_size=(img_rows, img_cols)
)""",language="python")
    st.info(f"**Output:** Found **{test_gen.n}** images in **{test_gen.num_classes}** classes → `{test_gen.class_indices}`")

    # Task 3
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown('<div class="task-hdr"><span class="task-num">✅ Task 3 · Length of train_generator</span></div>',unsafe_allow_html=True)
    st.code("print(len(train_generator))", language="python")
    st.success(f"**Output:** `{len(train_gen)}`")

    # Task 4 — build model and show summary
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown('<div class="task-hdr"><span class="task-num">✅ Task 4 · Model Summary</span></div>',unsafe_allow_html=True)
    st.code("model.summary()", language="python")
    with st.spinner("Building VGG16 model…"):
        model, bm = build_model(fine_tune=False)
    buf = io.StringIO()
    model.summary(print_fn=lambda x: buf.write(x+"\n"))
    st.code(buf.getvalue(), language="text")

    # Task 5 — compile
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown('<div class="task-hdr"><span class="task-num">✅ Task 5 · Compile the Model</span></div>',unsafe_allow_html=True)
    st.code("""model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    metrics=['accuracy']
)""", language="python")
    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  metrics=["accuracy"])
    st.success("✅ Model compiled successfully!")

    # Training — threaded so Stop button stays live
    import threading, time as _time
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown("### 🚀 Train Both Models")

    # Init session flags (only if not already set)
    if "training_active" not in st.session_state:
        st.session_state["training_active"] = False
    if "stop_requested" not in st.session_state:
        st.session_state["stop_requested"] = False
    if "train_progress" not in st.session_state:
        st.session_state["train_progress"] = {}

    def run_training(stop_event):
        """Background thread — uses its own fresh generators, never shares the UI cache."""
        import tensorflow as tf2
        from tensorflow.keras import optimizers
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.applications import vgg16
        from tensorflow.keras.models import Sequential, Model
        from tensorflow.keras.layers import Dense, Dropout, Flatten

        tr_idg = ImageDataGenerator(validation_split=VS, rescale=1/255.,
                                    width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        vl_idg = ImageDataGenerator(validation_split=VS, rescale=1/255.)
        tg = tr_idg.flow_from_directory(TRAIN_PATH, seed=SEED, batch_size=BS, class_mode="binary",
                                        shuffle=True, target_size=(IMG_SZ,IMG_SZ), subset="training")
        vg = vl_idg.flow_from_directory(TRAIN_PATH, seed=SEED, batch_size=BS, class_mode="binary",
                                        shuffle=True, target_size=(IMG_SZ,IMG_SZ), subset="validation")

        def make_fresh_model(fine_tune=False):
            vgg_base = vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(IMG_SZ,IMG_SZ,3))
            flat = Flatten()(vgg_base.layers[-1].output)
            bm   = Model(vgg_base.input, flat)
            for l in bm.layers: l.trainable = False
            if fine_tune:
                s = False
                for l in bm.layers:
                    if l.name == "block5_conv3": s = True
                    l.trainable = s
            return Sequential([bm, Dense(512,activation="relu"), Dropout(0.3),
                                Dense(512,activation="relu"), Dropout(0.3), Dense(1,activation="sigmoid")])

        class CB(tf2.keras.callbacks.Callback):
            def __init__(self, phase, total):
                self.phase = phase; self.total = total
            def on_epoch_end(self, e, logs={}):
                st.session_state["train_progress"] = {
                    "phase":   self.phase, "epoch": e+1, "total": self.total,
                    "acc":     logs.get("accuracy", 0), "val_acc": logs.get("val_accuracy", 0),
                    "loss":    logs.get("loss", 0),     "val_loss": logs.get("val_loss", 0),
                    "pct":     int(((e+1)/self.total)*50) if self.phase==1
                               else 50+int(((e+1)/self.total)*50),
                    "done": False
                }
                if stop_event.is_set():
                    self.model.stop_training = True

        try:
            m1 = make_fresh_model(fine_tune=False)
            m1.compile(loss="binary_crossentropy",
                       optimizer=tf2.keras.optimizers.Adam(learning_rate=1e-5), metrics=["accuracy"])
            h1 = m1.fit(tg, steps_per_epoch=5, epochs=10,
                        callbacks=[CB(1,10), LearningRateScheduler(exp_decay),
                                   EarlyStopping(monitor="val_loss",patience=4,min_delta=0.01),
                                   ModelCheckpoint(MODEL1,monitor="val_loss",save_best_only=True)],
                        validation_data=vg, validation_steps=vg.samples//BS, verbose=0)
            st.session_state["h1"] = h1.history

            if not stop_event.is_set():
                m2 = make_fresh_model(fine_tune=True)
                m2.compile(loss="binary_crossentropy",
                           optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=["accuracy"])
                h2 = m2.fit(tg, steps_per_epoch=5, epochs=10,
                            callbacks=[CB(2,10), LearningRateScheduler(exp_decay),
                                       EarlyStopping(monitor="val_loss",patience=4,min_delta=0.01),
                                       ModelCheckpoint(MODEL2,monitor="val_loss",save_best_only=True)],
                            validation_data=vg, validation_steps=vg.samples//BS, verbose=0)
                st.session_state["h2"] = h2.history
        finally:
            st.session_state["training_active"] = False
            st.session_state["train_progress"]["done"] = True
            # Clear UI caches so Tasks 1-5 re-renders fresh on next visit
            get_generators.clear()
            load_models.clear()


    if not st.session_state["training_active"]:
        if os.path.exists(MODEL1) and os.path.exists(MODEL2):
            st.success("✅ Models already trained and saved. Navigate to **Tasks 6–8** or **Tasks 9–10**.")
        st.info("Click **▶ Start Training Now** to run both phases. A **⏹ Stop Training** button will appear immediately.")
        if st.button("▶ Start Training Now", key="start_train"):
            stop_evt = threading.Event()
            st.session_state["stop_event"]       = stop_evt
            st.session_state["training_active"]  = True
            st.session_state["stop_requested"]   = False
            st.session_state["_done_rerun"]       = False
            st.session_state["train_progress"]   = {"phase":1,"epoch":0,"total":10,"acc":0,"val_acc":0,"loss":0,"val_loss":0,"pct":0,"done":False}
            t = threading.Thread(target=run_training, args=(stop_evt,), daemon=True)
            t.start()
            st.rerun()


    @st.fragment(run_every=2)
    def training_progress_panel():
        p      = st.session_state.get("train_progress", {})
        active = st.session_state.get("training_active", False)

        if active:
            pct = p.get("pct", 0)
            ph  = p.get("phase", 1)
            ep  = p.get("epoch", 0)
            phase_label = "Phase 1 · Extract Features" if ph == 1 else "Phase 2 · Fine-Tuning"

            # Live progress bar — smooth, no flicker
            st.progress(pct / 100, f"{phase_label} — Epoch {ep} / 10")

            # Live metric tiles
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Phase",     f"{ph} / 2")
            c2.metric("Epoch",     f"{ep} / 10")
            c3.metric("Train Acc", f"{p.get('acc', 0):.3f}")
            c4.metric("Val Acc",   f"{p.get('val_acc', 0):.3f}")

            # Stop button — always visible during training
            if not st.session_state.get("stop_requested"):
                if st.button("⏹ Stop Training", key="stop_btn_frag", type="primary"):
                    st.session_state.get("stop_event", threading.Event()).set()
                    st.session_state["stop_requested"] = True
            else:
                st.warning("🛑 Stop requested — finishing current epoch, please wait…")
                # Still show current metrics while waiting for epoch to end
                st.caption(f"Last known: Epoch {ep}/10 · acc {p.get('acc',0):.3f} · val_acc {p.get('val_acc',0):.3f}")

        elif p.get("done"):
            # Training just finished — do one clean full-page rerun to bring back the Start button
            if not st.session_state.get("_done_rerun"):
                st.session_state["_done_rerun"] = True
                st.rerun()
            if st.session_state.get("stop_requested"):
                st.warning("🛑 Training stopped early by user. Completed phases were saved.")
            # Success message shown by the outer block after rerun

    training_progress_panel()







# ══════════════════ TASKS 6-8 ═════════════════════════════════════════════════
elif page=="📈 Tasks 6–8: Training Curves":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st.markdown("## 📈 Tasks 6–8: Training & Validation Curves")
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)

    # Accept from session state OR reconstruct from saved models if refreshed
    h1 = st.session_state.get("h1")
    h2 = st.session_state.get("h2")

    if h1 is None or h2 is None:
        st.warning("⚠️ No training history found in this session. Please go to **Tasks 1–5** and click **Start Training**.")
        st.stop()

    def plot_curves(ep, y1, y2, l1, l2, title, ylabel, c1="#6fffe9", c2="#5bc0be"):
        fig, ax = plt.subplots(figsize=(7,4))
        fig.patch.set_facecolor("#0d1b2a"); ax.set_facecolor("#132030")
        ax.plot(ep,y1,color=c1,linewidth=2.5,marker="o",markersize=5,label=l1)
        ax.plot(ep,y2,color=c2,linewidth=2.5,marker="s",markersize=5,linestyle="--",label=l2)
        ax.set_title(title,color="#6fffe9",fontsize=13,fontweight="bold")
        ax.set_xlabel("Epochs",color="#8d99ae"); ax.set_ylabel(ylabel,color="#8d99ae")
        ax.tick_params(colors="#8d99ae")
        for sp in ax.spines.values(): sp.set_edgecolor("rgba(91,192,190,0.2)")
        ax.legend(facecolor="#0d2137",edgecolor="#5bc0be",labelcolor="#e0e1dd")
        ax.grid(True,alpha=0.12,color="#5bc0be")
        plt.tight_layout(); return fig

    ep1 = range(1,len(h1["loss"])+1)
    ep2 = range(1,len(h2["loss"])+1)

    st.markdown('<div class="task-hdr"><span class="task-num">✅ Task 6 · Accuracy Curves – Extract Features Model</span></div>',unsafe_allow_html=True)
    st.pyplot(plot_curves(ep1,h1["accuracy"],h1["val_accuracy"],"Training Accuracy","Validation Accuracy","Accuracy Curve (Extract Features)","Accuracy"))

    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown('<div class="task-hdr"><span class="task-num">✅ Task 7 · Loss Curves – Fine-Tune Model</span></div>',unsafe_allow_html=True)
    st.pyplot(plot_curves(ep2,h2["loss"],h2["val_loss"],"Training Loss","Validation Loss","Loss Curve (Fine-Tuned)","Loss","#e67e22","#f39c12"))

    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown('<div class="task-hdr"><span class="task-num">✅ Task 8 · Accuracy Curves – Fine-Tune Model</span></div>',unsafe_allow_html=True)
    st.pyplot(plot_curves(ep2,h2["accuracy"],h2["val_accuracy"],"Training Accuracy","Validation Accuracy","Accuracy Curve (Fine-Tuned)","Accuracy","#2ecc71","#27ae60"))

# ══════════════════ TASKS 9-10 ════════════════════════════════════════════════
elif page=="🖼️ Tasks 9–10: Predictions":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st.markdown("## 🖼️ Tasks 9–10: Visualising Test Predictions")
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)

    if not os.path.exists(MODEL1) or not os.path.exists(MODEL2):
        st.warning("⚠️ Saved models not found. Please go to **Tasks 1–5** and click **Start Training**.")
        st.stop()
    if not os.path.exists(DATA_DIR):
        st.warning("⚠️ Dataset not found. Please go to **Tasks 1–5** to download the data.")
        st.stop()

    tf2,*_ = get_tf()
    with st.spinner("Loading models…"):
        ef_model, ft_model = load_models()

    test_files_O = sorted(glob.glob(os.path.join(TEST_PATH,"O","*")))
    test_files_R = sorted(glob.glob(os.path.join(TEST_PATH,"R","*")))
    test_files   = test_files_O[:50]+test_files_R[:50]
    test_imgs    = np.array([tf2.keras.preprocessing.image.img_to_array(
                                 tf2.keras.preprocessing.image.load_img(f,target_size=(150,150)))
                             for f in test_files]).astype("float32")/255.
    test_labels  = [Path(f).parent.name for f in test_files]

    pred_ef = ["O" if p<0.5 else "R" for p in ef_model.predict(test_imgs,verbose=0)]
    pred_ft = ["O" if p<0.5 else "R" for p in ft_model.predict(test_imgs,verbose=0)]
    INDEX = 1

    def render(img_arr, model_name, actual, predicted):
        badge = "badge-r" if predicted=="R" else "badge-o"
        label = "Recyclable ♻️" if predicted=="R" else "Organic 🌱"
        correct = actual==predicted
        fig,ax = plt.subplots(figsize=(4,4))
        fig.patch.set_facecolor("#0d1b2a"); ax.set_facecolor("#0d1b2a")
        ax.imshow((img_arr*255).astype("uint8"))
        ax.set_title(f"Actual: {actual}  |  Predicted: {predicted}",color="#6fffe9",fontsize=11,fontweight="bold",pad=8)
        ax.axis("off"); plt.tight_layout()
        c1,c2 = st.columns([1,1.4])
        with c1: st.pyplot(fig)
        with c2:
            st.markdown(f"**Model:** `{model_name}`")
            st.markdown(f"**Actual Label:** `{actual}`")
            st.markdown(f"**Prediction:** &nbsp;<span class='{badge}'>{label}</span>",unsafe_allow_html=True)
            st.markdown(f"**Verdict:** {'✅ Correct' if correct else '⚠️ Incorrect (acceptable — stochastic training)'}")

    st.markdown(f'<div class="task-hdr"><span class="task-num">✅ Task 9 · Extract Features Model &nbsp;(index_to_plot = {INDEX})</span></div>',unsafe_allow_html=True)
    render(test_imgs[INDEX],"Extract Features Model",test_labels[INDEX],pred_ef[INDEX])

    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="task-hdr"><span class="task-num">✅ Task 10 · Fine-Tuned Model &nbsp;(index_to_plot = {INDEX})</span></div>',unsafe_allow_html=True)
    render(test_imgs[INDEX],"Fine-Tuned Model",test_labels[INDEX],pred_ft[INDEX])
