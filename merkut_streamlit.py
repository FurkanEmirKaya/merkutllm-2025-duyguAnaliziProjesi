import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_path = ".\\checkpoint-1860"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


st.set_page_config(page_title="Duygu Analizi Uygulaması", page_icon="💬", layout="wide")


st.title("💬 Merküt - Destan Türkçe Metin Duygu Analizi")
st.markdown(
    """
    Bu uygulama, Türkçe metinlerde duygusal içerikleri analiz eder ve 5 farklı sınıftan hangisine dahil olduğunu tespit eder.
    Aşağıdaki alana analiz edilecek cümleyi girin.
    
    - **HAKARET** \n - **IRKÇILIK** \n - **CİNSİYETÇİLİK** \n - **KÜFÜR** \n - **DİĞER**
    """
)


col1, col2 = st.columns([3, 1])


label_mapping = {
    0: "HAKARET",
    1: "IRKÇILIK",
    2: "CİNSİYETÇİLİK",
    3: "KÜFÜR",
    4: "DİĞER"
}

is_offensive = {
    0: "True",
    1: "True",
    2: "True",
    3: "True",
    4: "False" # İleride detaylandırılabilir. Övgü, tavsiye
}


def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax().item()

    predicted_label = label_mapping[predicted_class]
    offensive_status = is_offensive[predicted_class]

    return predicted_label, offensive_status


with col1:
    user_input = st.text_area("Lütfen sınıflandırılacak metni girin:", height=150, placeholder="Metninizi buraya yazın...")

    if user_input:
        with st.spinner("Tahmin yapılıyor..."):
            predicted_label, offensive_status = predict(user_input)
        st.success(f"Tahmin edilen sınıf: **{predicted_label}**", icon="✅")
        st.info(f"Saldırganlık durumu: **{offensive_status}**")

with col2:
    st.subheader("Örnek Metinler")
    st.info("Bir metni denemek ve sınıf türlerini görmek için aşağıdaki seçenekleri deneyin.")

    examples = [
        "Kadın aklıyla yola düşülmez.",
        "Bu çok aşağılayıcı bir cümle.",
        "Kürt gibi konuşma!",
        "Siktir git!",
        "Sen akılsız bir salaksın."
    ]

    for example in examples:
        if st.button(f"➡️ {example}"):
            with st.spinner("Analiz ediliyor..."):
                predicted_label, offensive_status = predict(example)
            st.write(f"**Tahmin edilen sınıf:** {predicted_label}")
            st.write(f"**Saldırganlık durumu:** {offensive_status}")

st.markdown("---")
st.caption("© 2025 Duygu Analizi | Gazi Merküt Doğal Dil İşleme Takımı")
