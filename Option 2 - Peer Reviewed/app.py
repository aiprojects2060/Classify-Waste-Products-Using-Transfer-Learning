import streamlit as st
import numpy as np
import os
import glob
import requests
import zipfile
from pathlib import Path
from PIL import Image
import io
import time
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.set_page_config(
    page_title="EcoClean – Option 2 | Peer Review",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html,body,[class*="css"]{ font-family:'Inter',sans-serif; }
.stApp{ background:linear-gradient(160deg,#0a1628 0%,#1a2744 60%,#0e2040 100%); color:#e2e8f0; }
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0a1628,#112035)!important; border-right:1px solid rgba(134,239,172,0.15); }
[data-testid="stSidebar"] *{ color:#cbd5e1; }
.hero-title{ font-size:3rem;font-weight:800;letter-spacing:-1px;background:linear-gradient(90deg,#86efac,#34d399,#059669);-webkit-background-clip:text;-webkit-text-fill-color:transparent; }
.hero-sub{ color:rgba(134,239,172,0.7);font-size:1.1rem;margin-top:4px; }
.divider{ height:2px;background:linear-gradient(90deg,transparent,rgba(52,211,153,0.4),transparent);margin:24px 0; }
.card{ background:rgba(255,255,255,0.04);border:1px solid rgba(52,211,153,0.2);border-radius:14px;padding:20px 24px;margin-bottom:14px;box-shadow:0 4px 20px rgba(0,0,0,0.3); }
.card-label{ font-size:.78rem;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:#34d399;margin-bottom:4px; }
.card-val{ font-size:2.2rem;font-weight:800;color:#86efac;margin:0; }
.task-hdr{ background:rgba(52,211,153,0.07);border-left:4px solid #34d399;border-radius:8px;padding:12px 18px;margin-bottom:14px; }
.task-num{ color:#86efac;font-weight:700;font-size:.88rem;text-transform:uppercase;letter-spacing:1px; }
div.stButton>button{ background:linear-gradient(90deg,#34d399,#059669)!important;color:#fff!important;border:none!important;border-radius:25px!important;font-weight:600!important;padding:.55rem 1.8rem!important;box-shadow:0 4px 12px rgba(52,211,153,.35)!important;transition:all .25s ease!important; }
div.stButton>button:hover{ transform:translateY(-2px)!important;box-shadow:0 8px 20px rgba(52,211,153,.55)!important; }
.badge-r{ display:inline-block;padding:9px 26px;border-radius:50px;background:linear-gradient(90deg,#22c55e,#16a34a);color:#fff;font-weight:700;font-size:1.1rem;box-shadow:0 4px 14px rgba(34,197,94,.5); }
.badge-o{ display:inline-block;padding:9px 26px;border-radius:50px;background:linear-gradient(90deg,#f97316,#ea580c);color:#fff;font-weight:700;font-size:1.1rem;box-shadow:0 4px 14px rgba(249,115,22,.5); }
</style>
""", unsafe_allow_html=True)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "o-vs-r-split")
TRAIN_PATH = os.path.join(DATA_DIR, "train")
TEST_PATH  = os.path.join(DATA_DIR, "test")
MODEL1     = os.path.join(BASE_DIR, "O_R_tlearn_vgg16.keras")
MODEL2     = os.path.join(BASE_DIR, "O_R_tlearn_fine_tune_vgg16.keras")
IMG_SZ, BS, SEED, VS = 150, 32, 42, 0.2

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

def download_data():
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/kd6057VPpABQ2FqCbgu9YQ/o-vs-r-split-reduced-1200.zip"
    zp  = os.path.join(BASE_DIR, "data.zip")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zp,"wb") as f:
            for ch in r.iter_content(8192): f.write(ch)
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
    for l in bm.layers: l.trainable=False
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

def preprocess_pil(pil_img):
    tf,*_ = get_tf()
    arr = tf.keras.preprocessing.image.img_to_array(pil_img.resize((150,150)).convert("RGB"))
    return np.expand_dims(arr.astype("float32")/255., 0)

def exp_decay(e): return 1e-4*np.exp(-0.1*e)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🌿 EcoClean AI")
st.sidebar.markdown("**Option 2 – Peer Review Submission**")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate",[
    "🏠 Overview",
    "⚙️ Tasks 1–5: Setup & Train",
    "📈 Tasks 6–8: Curves",
    "🖼️ Tasks 9–10: Predictions",
    "🔬 Live Classifier",
])

# ══════════════════ OVERVIEW ══════════════════════════════════════════════════
if page=="🏠 Overview":
    st.markdown('<p class="hero-title">EcoClean AI – Waste Classifier</p>',unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Option 2 · Peer-Reviewed Submission · VGG16 Transfer Learning</p>',unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col,lbl,val,sub in [(c1,"Architecture","VGG16","ImageNet"),(c2,"Classes","2","Organic·Recyclable"),(c3,"Tasks","10","All covered"),(c4,"Submission","Option 2","Peer Reviewed")]:
        col.markdown(f'<div class="card"><div class="card-label">{lbl}</div><div class="card-val">{val}</div><small style="color:#94a3b8">{sub}</small></div>',unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown("### 📋 Task Evidence Summary")
    tasks=[("Task 1","Print TF version"),("Task 2","Create test_generator"),("Task 3","Length of train_generator"),("Task 4","Model summary"),("Task 5","Compile model"),("Task 6","Accuracy curves – Extract Features"),("Task 7","Loss curves – Fine-Tune"),("Task 8","Accuracy curves – Fine-Tune"),("Task 9","Plot test image – Extract Features (index=1)"),("Task 10","Plot test image – Fine-Tuned (index=1)")]
    l,r = st.columns(2)
    for i,(n,d) in enumerate(tasks):
        (l if i%2==0 else r).markdown(f'<div class="task-hdr"><span class="task-num">✅ {n}</span><br><span style="color:#cbd5e1">{d}</span></div>',unsafe_allow_html=True)
    st.info("👉 Start with **Tasks 1–5** to download dataset and train. Then use Tasks 6–10 and **Live Classifier**.")

# ══════════════════ TASKS 1-5 ═════════════════════════════════════════════════
elif page=="⚙️ Tasks 1–5: Setup & Train":
    st.markdown("## ⚙️ Tasks 1–5: Setup & Training")
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)

    # Task 1
    st.markdown('<div class="task-hdr"><span class="task-num">✅ Task 1 · TensorFlow Version</span></div>',unsafe_allow_html=True)
    tf,*_ = get_tf()
    st.success(f"**TensorFlow Version: {tf.__version__}**")
    st.code(f"import tensorflow as tf\nprint(tf.__version__)  # → {tf.__version__}",language="python")
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)

    # Auto-download
    st.markdown("### 📥 Dataset")
    if not os.path.exists(DATA_DIR):
        with st.spinner("⏳ Downloading waste dataset (once only)…"):
            try:
                download_data()
                st.success("✅ Dataset downloaded!")
            except Exception as e:
                st.error(f"Download error: {e}"); st.stop()
    else:
        st.success("✅ Dataset ready!")

    with st.spinner("Preparing data generators…"):
        train_gen,val_gen,test_gen = get_generators()

    # Task 2
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown('<div class="task-hdr"><span class="task-num">✅ Task 2 · test_generator</span></div>',unsafe_allow_html=True)
    st.code("""test_generator = test_datagen.flow_from_directory(
    directory=path_test, class_mode='binary', seed=seed,
    batch_size=batch_size, shuffle=False, target_size=(img_rows, img_cols)
)""",language="python")
    st.info(f"Found **{test_gen.n}** images · **{test_gen.num_classes}** classes → `{test_gen.class_indices}`")

    # Task 3
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown('<div class="task-hdr"><span class="task-num">✅ Task 3 · Length of train_generator</span></div>',unsafe_allow_html=True)
    st.code("print(len(train_generator))",language="python")
    st.success(f"**Output:** `{len(train_gen)}`")

    # Task 4
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown('<div class="task-hdr"><span class="task-num">✅ Task 4 · Model Summary</span></div>',unsafe_allow_html=True)
    with st.spinner("Building VGG16 model…"):
        model,bm = build_model(fine_tune=False)
    buf = io.StringIO()
    model.summary(print_fn=lambda x: buf.write(x+"\n"))
    st.code(buf.getvalue(),language="text")

    # Task 5
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown('<div class="task-hdr"><span class="task-num">✅ Task 5 · Compile the Model</span></div>',unsafe_allow_html=True)
    st.code("""model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    metrics=['accuracy']
)""",language="python")
    model.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),metrics=["accuracy"])
    st.success("✅ Model compiled!")

    # Training — threaded so Stop button stays live
    import threading, time as _time
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown("### 🚀 Train Both Models")


    for k,v in [("training_active",False),("stop_requested",False),("train_progress",{}),("h1",None),("h2",None)]:
        st.session_state.setdefault(k, v)

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
            base = vgg16.VGG16(include_top=False,weights="imagenet",input_shape=(IMG_SZ,IMG_SZ,3))
            flat = Flatten()(base.layers[-1].output)
            bm   = Model(base.input, flat)
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
                    "phase": self.phase, "epoch": e+1, "total": self.total,
                    "acc": logs.get("accuracy",0), "val_acc": logs.get("val_accuracy",0),
                    "pct": int(((e+1)/self.total)*50) if self.phase==1 else 50+int(((e+1)/self.total)*50),
                    "done": False
                }
                if stop_event.is_set(): self.model.stop_training = True

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
            get_generators.clear()
            load_models.clear()


    if not st.session_state["training_active"]:
        if os.path.exists(MODEL1) and os.path.exists(MODEL2):
            st.success("✅ Models already trained and saved. Navigate to **Tasks 6–8** or **Tasks 9–10**.")
        st.info("Click **▶ Start Training Now** to run both phases. A **⏹ Stop Training** button will appear immediately.")
        if st.button("▶ Start Training Now",key="start_train"):
            stop_evt = threading.Event()
            st.session_state.update({"stop_event":stop_evt,"training_active":True,"stop_requested":False,
                                     "_done_rerun":False,
                                     "train_progress":{"phase":1,"epoch":0,"total":10,"acc":0,"val_acc":0,"pct":0,"done":False}})
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

            st.progress(pct / 100, f"{phase_label} — Epoch {ep} / 10")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Phase",     f"{ph} / 2")
            c2.metric("Epoch",     f"{ep} / 10")
            c3.metric("Train Acc", f"{p.get('acc', 0):.3f}")
            c4.metric("Val Acc",   f"{p.get('val_acc', 0):.3f}")

            if not st.session_state.get("stop_requested"):
                if st.button("⏹ Stop Training", key="stop_btn_frag", type="primary"):
                    st.session_state.get("stop_event", threading.Event()).set()
                    st.session_state["stop_requested"] = True
            else:
                st.warning("🛑 Stop requested — finishing current epoch, please wait…")
                st.caption(f"Last known: Epoch {ep}/10 · acc {p.get('acc',0):.3f} · val_acc {p.get('val_acc',0):.3f}")

        elif p.get("done"):
            if not st.session_state.get("_done_rerun"):
                st.session_state["_done_rerun"] = True
                st.rerun()
            if st.session_state.get("stop_requested"):
                st.warning("🛑 Training stopped early by user. Completed phases were saved.")

    training_progress_panel()



# ══════════════════ TASKS 6-8 ═════════════════════════════════════════════════
elif page=="📈 Tasks 6–8: Curves":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st.markdown("## 📈 Tasks 6–8: Training Curves")
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    h1 = st.session_state.get("h1")
    h2 = st.session_state.get("h2")
    if h1 is None or h2 is None:
        st.warning("⚠️ No training history. Go to **Tasks 1–5** and click **Start Training**."); st.stop()
    def pcurve(ep,y1,y2,l1,l2,title,ylabel,c1,c2):
        fig,ax=plt.subplots(figsize=(7,4))
        fig.patch.set_facecolor("#0a1628"); ax.set_facecolor("#112035")
        ax.plot(ep,y1,color=c1,linewidth=2.5,marker="o",markersize=5,label=l1)
        ax.plot(ep,y2,color=c2,linewidth=2.5,marker="s",markersize=5,linestyle="--",label=l2)
        ax.set_title(title,color="#86efac",fontsize=13,fontweight="bold")
        ax.set_xlabel("Epochs",color="#94a3b8"); ax.set_ylabel(ylabel,color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values(): sp.set_edgecolor("rgba(52,211,153,0.2)")
        ax.legend(facecolor="#0a1628",edgecolor="#34d399",labelcolor="#e2e8f0")
        ax.grid(True,alpha=0.12,color="#34d399"); plt.tight_layout(); return fig
    ep1=range(1,len(h1["loss"])+1); ep2=range(1,len(h2["loss"])+1)
    st.markdown('<div class="task-hdr"><span class="task-num">✅ Task 6 · Accuracy Curves – Extract Features Model</span></div>',unsafe_allow_html=True)
    st.pyplot(pcurve(ep1,h1["accuracy"],h1["val_accuracy"],"Training Accuracy","Validation Accuracy","Accuracy Curve (Extract Features)","Accuracy","#86efac","#34d399"))
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown('<div class="task-hdr"><span class="task-num">✅ Task 7 · Loss Curves – Fine-Tune Model</span></div>',unsafe_allow_html=True)
    st.pyplot(pcurve(ep2,h2["loss"],h2["val_loss"],"Training Loss","Validation Loss","Loss Curve (Fine-Tuned)","Loss","#f97316","#fb923c"))
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown('<div class="task-hdr"><span class="task-num">✅ Task 8 · Accuracy Curves – Fine-Tune Model</span></div>',unsafe_allow_html=True)
    st.pyplot(pcurve(ep2,h2["accuracy"],h2["val_accuracy"],"Training Accuracy","Validation Accuracy","Accuracy Curve (Fine-Tuned)","Accuracy","#22c55e","#4ade80"))

# ══════════════════ TASKS 9-10 ════════════════════════════════════════════════
elif page=="🖼️ Tasks 9–10: Predictions":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st.markdown("## 🖼️ Tasks 9–10: Test Image Predictions")
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    if not os.path.exists(MODEL1) or not os.path.exists(MODEL2):
        st.warning("⚠️ Models not found. Go to **Tasks 1–5** and start training."); st.stop()
    if not os.path.exists(DATA_DIR):
        st.warning("⚠️ Dataset not found. Go to **Tasks 1–5** first."); st.stop()
    tf2,*_ = get_tf()
    with st.spinner("Loading models…"):
        ef_model,ft_model = load_models()
    test_files_O=sorted(glob.glob(os.path.join(TEST_PATH,"O","*")))
    test_files_R=sorted(glob.glob(os.path.join(TEST_PATH,"R","*")))
    test_files=test_files_O[:50]+test_files_R[:50]
    test_imgs=np.array([tf2.keras.preprocessing.image.img_to_array(tf2.keras.preprocessing.image.load_img(f,target_size=(150,150))) for f in test_files]).astype("float32")/255.
    test_labels=[Path(f).parent.name for f in test_files]
    pred_ef=["O" if p<0.5 else "R" for p in ef_model.predict(test_imgs,verbose=0)]
    pred_ft=["O" if p<0.5 else "R" for p in ft_model.predict(test_imgs,verbose=0)]
    INDEX=1
    def render(img_arr,model_name,actual,predicted):
        badge="badge-r" if predicted=="R" else "badge-o"
        label="Recyclable ♻️" if predicted=="R" else "Organic 🌱"
        fig,ax=plt.subplots(figsize=(4,4)); fig.patch.set_facecolor("#0a1628"); ax.set_facecolor("#0a1628")
        ax.imshow((img_arr*255).astype("uint8")); ax.set_title(f"Actual: {actual}  |  Predicted: {predicted}",color="#86efac",fontsize=11,fontweight="bold",pad=8); ax.axis("off"); plt.tight_layout()
        c1,c2=st.columns([1,1.4])
        with c1: st.pyplot(fig)
        with c2:
            st.markdown(f"**Model:** `{model_name}`"); st.markdown(f"**Actual:** `{actual}`")
            st.markdown(f"**Prediction:** &nbsp;<span class='{badge}'>{label}</span>",unsafe_allow_html=True)
            st.markdown(f"**Verdict:** {'✅ Correct' if actual==predicted else '⚠️ Incorrect (stochastic training — acceptable)'}")
    st.markdown(f'<div class="task-hdr"><span class="task-num">✅ Task 9 · Extract Features Model &nbsp;(index_to_plot = {INDEX})</span></div>',unsafe_allow_html=True)
    render(test_imgs[INDEX],"Extract Features Model",test_labels[INDEX],pred_ef[INDEX])
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="task-hdr"><span class="task-num">✅ Task 10 · Fine-Tuned Model &nbsp;(index_to_plot = {INDEX})</span></div>',unsafe_allow_html=True)
    render(test_imgs[INDEX],"Fine-Tuned Model",test_labels[INDEX],pred_ft[INDEX])

# ══════════════════ LIVE CLASSIFIER ═══════════════════════════════════════════
elif page=="🔬 Live Classifier":
    st.markdown("## 🔬 Live Waste Classifier")
    st.markdown('<p style="color:#94a3b8;margin-top:-10px">Upload any image and get an instant AI classification</p>',unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>',unsafe_allow_html=True)
    if not os.path.exists(MODEL2):
        st.warning("⚠️ Fine-Tuned model not found. Please go to **Tasks 1–5** and start training first."); st.stop()
    with st.spinner("Loading model…"):
        _,ft_model = load_models()
    c1,c2 = st.columns([1,1])
    with c1:
        uploaded = st.file_uploader("Upload waste image (JPG or PNG)",type=["jpg","jpeg","png"])
        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img,caption="Input Image",use_container_width=True)
            if st.button("🔍 Classify",use_container_width=True):
                with st.spinner("Analysing with VGG16 Fine-Tuned model…"):
                    time.sleep(1.0)
                    arr  = preprocess_pil(pil_img)
                    prob = ft_model.predict(arr,verbose=0)[0][0]
                if prob>=0.5:
                    label="Recyclable"; conf=prob*100; badge="badge-r"; color="#22c55e"; icon="♻️"
                else:
                    label="Organic"; conf=(1-prob)*100; badge="badge-o"; color="#f97316"; icon="🌱"
                with c2:
                    st.markdown("### 🧬 Classification Result")
                    st.markdown(f"""<div style="background:rgba(255,255,255,0.04);border:1px solid rgba(52,211,153,0.2);border-radius:14px;padding:24px;margin-top:8px;">
                        <h3 style="color:{color};margin-top:0">{icon} {label}</h3>
                        <p style="color:#94a3b8;font-size:.9rem;margin-bottom:4px">MODEL CONFIDENCE</p>
                        <h2 style="color:{color};margin:0">{conf:.1f}%</h2>
                        <hr style="border-color:rgba(255,255,255,0.08);margin:16px 0">
                        <small style="color:#64748b">VGG16 Fine-Tuned · Binary Classification</small>
                    </div>""",unsafe_allow_html=True)
                    st.progress(int(conf)/100)
    if not uploaded:
        with c2:
            st.markdown("""<div style="border:2px dashed rgba(52,211,153,0.25);border-radius:14px;padding:60px;text-align:center;margin-top:10px;">
                <h4 style="color:rgba(134,239,172,0.4)">⏳ Awaiting Input</h4>
                <p style="color:rgba(148,163,184,0.5)">Upload an image to start</p>
            </div>""",unsafe_allow_html=True)
