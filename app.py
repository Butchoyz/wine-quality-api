import streamlit as st
import joblib
import numpy as np
from streamlit.components.v1 import html

# Load model and imputers
model = joblib.load("xgb_wine_model.pkl")
imputer_mean = joblib.load("imputer_mean_density.pkl")
imputer_median = joblib.load("imputer_median_citric_pH.pkl")

# Configure page
st.set_page_config(
    page_title="Sanborn boutique wine: Wine Quality Predictor",
    page_icon="🍷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Enhanced Wine-themed styling with matching sidebar background
def set_wine_background():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255, 255, 255, 0.97), rgba(255, 255, 255, 0.97)), 
                        url('https://images.unsplash.com/photo-1559496417-e7f25cb247f3?q=80&w=2564');
            background-size: cover;
            background-attachment: fixed;
            background-position: center 30%;
        }}
        
        @media (prefers-color-scheme: dark) {{
            .stApp {{
                background: linear-gradient(rgba(10, 5, 7, 0.97), rgba(10, 5, 7, 0.97)), 
                            url('https://images.unsplash.com/photo-1559496417-e7f25cb247f3?q=80&w=2564');
                background-size: cover;
                background-attachment: fixed;
                background-position: center 30%;
            }}
        }}
        
        /* Sidebar background matching the main app */
        section[data-testid="stSidebar"] > div:first-child {{
            background: linear-gradient(rgba(255, 255, 255, 0.97), rgba(255, 255, 255, 0.97)), 
                        url('https://images.unsplash.com/photo-1559496417-e7f25cb247f3?q=80&w=2564') !important;
            background-size: cover !important;
            background-attachment: fixed !important;
            background-position: center 30% !important;
        }}
        
        @media (prefers-color-scheme: dark) {{
            section[data-testid="stSidebar"] > div:first-child {{
                background: linear-gradient(rgba(10, 5, 7, 0.97), rgba(10, 5, 7, 0.97)), 
                            url('https://images.unsplash.com/photo-1559496417-e7f25cb247f3?q=80&w=2564') !important;
            }}
        }}
        
        .header-container {{
            background: transparent;
            padding: 2rem 0 1rem;
            margin-bottom: 1rem;
            text-align: center;
            position: relative;
        }}
        
        .main-title {{
            color: #5c162e !important;
            font-family: 'Playfair Display', serif;
            font-weight: 800;
            font-size: 3.2rem;
            letter-spacing: 1px;
            margin-bottom: 0.5rem;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
        }}
        
        .subtitle {{
            color: #7d3a4e !important;
            font-family: 'Lora', serif;
            font-weight: 500;
            font-size: 1.3rem;
            letter-spacing: 1.5px;
            max-width: 700px;
            margin: 0 auto;
            padding: 0 20px;
        }}
        
        @media (prefers-color-scheme: dark) {{
            .main-title {{
                color: #f4e3d7 !important;
                text-shadow: 0 2px 8px rgba(0,0,0,0.5);
            }}
            .subtitle {{
                color: #d8b49d !important;
            }}
        }}
        
        .input-section {{
            background-color: rgba(255, 255, 255, 0.88);
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 8px 30px rgba(122, 21, 56, 0.08);
            border: 1px solid rgba(92, 22, 46, 0.15);
            backdrop-filter: blur(3px);
            position: relative;
            overflow: hidden;
        }}
        
        .input-section:before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #5c162e, #d4a798, #5c162e);
        }}
        
        @media (prefers-color-scheme: dark) {{
            .input-section {{
                background-color: rgba(25, 12, 16, 0.88);
                border: 1px solid rgba(109, 13, 44, 0.3);
                box-shadow: 0 8px 30px rgba(0,0,0,0.25);
            }}
        }}
        
        .feature-header {{
            font-family: 'Playfair Display', serif;
            color: #5c162e;
            padding: 15px 0 8px;
            margin-top: 15px;
            font-weight: 700;
            letter-spacing: 0.8px;
            position: relative;
            font-size: 1.4rem;
        }}
        
        .feature-header:after {{
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 80px;
            height: 2px;
            background: linear-gradient(90deg, #5c162e, #d4a798);
        }}
        
        @media (prefers-color-scheme: dark) {{
            .feature-header {{
                color: #f4e3d7;
            }}
            .feature-header:after {{
                background: linear-gradient(90deg, #a86e5a, #d4a798);
            }}
        }}
        
        .prediction-card {{
            border-radius: 18px;
            padding: 40px 35px;
            margin: 50px 0 30px;
            box-shadow: 0 12px 40px rgba(0,0,0,0.12);
            transition: all 0.5s ease;
            backdrop-filter: blur(5px);
            position: relative;
            overflow: hidden;
            border: none;
        }}
        
        .prediction-card:before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 6px;
            background: linear-gradient(90deg, #4caf50, #a5d6a7, #4caf50);
            z-index: 2;
        }}
        
        .bad-wine:before {{
            background: linear-gradient(90deg, #f44336, #ffab91, #f44336);
        }}
        
        .good-wine {{
            background: linear-gradient(135deg, rgba(232, 245, 233, 0.92) 0%, rgba(209, 232, 210, 0.92) 100%);
            animation: glow-good 3s ease-in-out infinite;
        }}
        
        .bad-wine {{
            background: linear-gradient(135deg, rgba(255, 235, 238, 0.92) 0%, rgba(255, 219, 222, 0.92) 100%);
            animation: glow-bad 3s ease-in-out infinite;
        }}
        
        @keyframes glow-good {{
            0% {{ box-shadow: 0 0 15px rgba(76, 175, 80, 0.4); }}
            50% {{ box-shadow: 0 0 35px rgba(76, 175, 80, 0.7); }}
            100% {{ box-shadow: 0 0 15px rgba(76, 175, 80, 0.4); }}
        }}
        
        @keyframes glow-bad {{
            0% {{ box-shadow: 0 0 15px rgba(244, 67, 54, 0.4); }}
            50% {{ box-shadow: 0 0 35px rgba(244, 67, 54, 0.7); }}
            100% {{ box-shadow: 0 0 15px rgba(244, 67, 54, 0.4); }}
        }}
        
        @media (prefers-color-scheme: dark) {{
            .good-wine {{
                background: linear-gradient(135deg, rgba(20, 45, 22, 0.9) 0%, rgba(10, 30, 12, 0.9) 100%);
            }}
            .bad-wine {{
                background: linear-gradient(135deg, rgba(58, 18, 21, 0.9) 0%, rgba(41, 9, 11, 0.9) 100%);
            }}
            .prediction-card h1, .prediction-card h2, .prediction-card h3, .prediction-card p {{
                color: #f0e0d8 !important;
            }}
        }}
        
        .stButton>button {{
            background: linear-gradient(135deg, #7a1538 0%, #5c162e 100%);
            color: white !important;
            border-radius: 8px !important;
            padding: 16px 32px;
            font-weight: 600;
            font-family: 'Playfair Display', serif;
            font-size: 1.2rem;
            border: none;
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 30px;
            letter-spacing: 1.5px;
            position: relative;
            overflow: hidden;
            z-index: 1;
        }}
        
        .stButton>button:before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: 0.5s;
            z-index: -1;
        }}
        
        .stButton>button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 12px 25px rgba(122, 21, 56, 0.35);
        }}
        
        .stButton>button:hover:before {{
            left: 100%;
        }}
        
        .stNumberInput input {{
            background-color: #faf5f3 !important;
            border-radius: 6px !important;
            border: 1px solid #e0cbc0 !important;
            padding: 10px 12px !important;
            font-family: 'Lora', serif;
        }}
        
        @media (prefers-color-scheme: dark) {{
            .stNumberInput input {{
                background-color: rgba(35, 15, 20, 0.9) !important;
                color: #f0e0d8 !important;
                border: 1px solid #5c3a31 !important;
            }}
        }}
        
        .confidence-meter {{
            margin: 30px 0;
            padding: 0;
        }}
        
        .wine-tip {{
            background: rgba(109, 13, 44, 0.07);
            border-radius: 10px;
            border-left: 4px solid #7a1538;
            padding: 20px;
            margin: 25px 0;
            font-family: 'Lora', serif;
            font-size: 1.05rem;
        }}
        
        footer {{
            text-align: center;
            padding: 40px 20px 30px;
            color: #7d3a4e;
            font-size: 0.95rem;
            margin-top: 40px;
            font-family: 'Lora', serif;
            position: relative;
        }}
        
        footer:before {{
            content: '';
            position: absolute;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            width: 100px;
            height: 2px;
            background: linear-gradient(90deg, transparent, #d4a798, transparent);
        }}
        
        .performance-badge {{
            display: inline-block;
            background: rgba(122, 21, 56, 0.12);
            padding: 8px 18px;
            border-radius: 25px;
            font-size: 0.9rem;
            margin: 8px;
            border: 1px solid rgba(122, 21, 56, 0.15);
            font-family: 'Lora', serif;
        }}
        
        .vine-icon {{
            font-size: 1.8rem;
            margin: 0 5px;
            vertical-align: middle;
        }}
        
        .floating-grapes {{
            position: absolute;
            opacity: 0.05;
            z-index: -1;
            font-size: 8rem;
            transform: rotate(-25deg);
        }}
        
        .grapes-left {{
            top: 20%;
            left: -30px;
        }}
        
        .grapes-right {{
            bottom: 20%;
            right: -30px;
            transform: rotate(25deg);
        }}
        </style>
        
        <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800;900&family=Lora:wght@400;500;600;700&display=swap" rel="stylesheet">
        """,
        unsafe_allow_html=True
    )

set_wine_background()



# Sidebar - Sommelier's journal
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; margin-bottom:25px; padding-top:10px;'>
        <h1 style='font-family: "Playfair Display", serif; font-size: 2rem; 
        color: #5c162e; border-bottom: 2px solid #d4a798; 
        padding-bottom: 15px; margin-bottom: 20px;'>Sanborn boutique wine company</h1>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Tasting Notes", expanded=True):
        st.markdown("""
        **Key Parameters:**
        - **Fixed Acidity**: 4-10 g/dm³ (tartaric acid)  
        - **Volatile Acidity**: <0.5 g/dm³ (acetic acid)  
        - **Citric Acid**: 0.2-0.5 g/dm³ (freshness)  
        - **Residual Sugar**: 1-15 g/dm³ (sweetness)  
        """)
        
        st.markdown("""
        <div style="background: rgba(122, 21, 56, 0.08); padding: 15px; border-radius: 10px; margin: 15px 0;">
        <p style="font-weight:600; color:#5c162e; margin-bottom:8px;">Quality Thresholds</p>
        <p>Premium wines typically have:</p>
        <ul style="margin-top:5px; padding-left:20px;">
            <li>Alcohol > 12% vol</li>
            <li>pH between 3.2-3.6</li>
            <li>Sulphates > 0.5 g/dm³</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with st.expander("Sanborn boutique wine Metrics"):
        st.markdown("""
        **Dataset:**  
        <span class="vine-icon">🍷</span> 1,599 curated wine samples  
        
        **Model Performance:**  
        <span class="vine-icon">⭐</span> 95% overall accuracy  
        <span class="vine-icon">🏆</span> 0.95 macro F1-score  
        """, unsafe_allow_html=True)
        
    st.markdown("""
    <style>
        .wine-bullets-container {
            background: rgba(122, 21, 56, 0.05);
            padding: 25px;
            border-radius: 12px;
            margin-top: 20px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
            font-family: 'Segoe UI', sans-serif;
        }

        .wine-bullets-title {
            font-weight: 700;
            color: #5c162e;
            font-size: 20px;
            margin-bottom: 15px;
        }

        .wine-bullet-list {
            list-style-type: disc;
            padding-left: 25px;
            color: #3e0f1c;
            font-size: 16px;
            line-height: 1.7;
        }

        .wine-bullet-list li::marker {
            color: #7a1538;
        }
    </style>

    <div class="wine-bullets-container">
        <div class="wine-bullets-title">📊 Classification Report</div>
        <ul class="wine-bullet-list">
            <li><strong>Class 0:</strong> Precision 0.98, Recall 0.93, F1 Score 0.95</li>
            <li><strong>Class 1:</strong> Precision 0.93, Recall 0.98, F1 Score 0.96</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

    st.divider()
    
    st.markdown("""
    <div style='background: rgba(122, 21, 56, 0.07); padding: 20px; border-radius: 10px; border-left: 3px solid #7a1538; margin-top: 20px;'>
        <p style='font-family: "Playfair Display", serif; font-style: italic; color:#5c162e; font-size: 1.1rem;'>
        "Great wine requires a mad man to grow the vine, a wise man to watch it, a lucid poet to make it, and a lover to drink it."
        </p>
        <p style='text-align: right; margin-top: 10px; color:#7d3a4e; font-weight: 500;'>— Salvador Dali</p>
    </div>
    """, unsafe_allow_html=True)

# Main content - Elegant header
st.markdown("""
<div class="header-container">
    <h1 class="main-title">Sanborn boutique wine</h1>
    <p class="subtitle">Premier Wine Quality Assurance System</p>
</div>
""", unsafe_allow_html=True)

# Feature input sections with refined layout
with st.form("wine_form"):
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        
        # Acidity Group
        st.markdown('<div class="feature-header">Acidity Profile</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            fixed_acidity = st.number_input("Fixed Acidity (g/dm³)", min_value=0.0, value=7.0, step=0.1,
                                           help="Tartaric and malic acids")
        with c2:
            volatile_acidity = st.number_input("Volatile Acidity", min_value=0.0, value=0.3, step=0.01,
                                              help="Acetic acid concentration")
        citric_acid = st.number_input("Citric Acid (g/dm³)", min_value=0.0, value=0.3, step=0.01,
                                      help="Freshness and preservative")
        
        # Sweetness & Sulfur
        st.markdown('<div class="feature-header">Sweetness & Sulfur</div>', unsafe_allow_html=True)
        residual_sugar = st.number_input("Residual Sugar (g/dm³)", min_value=0.0, value=5.0, step=0.1,
                                         help="Unfermented sugar content")
        
        c1, c2 = st.columns(2)
        with c1:
            free_sulfur_dioxide = st.number_input("Free SO₂ (mg/dm³)", min_value=0.0, value=25.0, step=1.0,
                                                 help="Antimicrobial protection")
        with c2:
            total_sulfur_dioxide = st.number_input("Total SO₂ (mg/dm³)", min_value=0.0, value=100.0, step=1.0,
                                                  help="Total sulfur content")
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close input-section container
    
    with col2:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        
        # Chemical Properties
        st.markdown('<div class="feature-header">Chemical Composition</div>', unsafe_allow_html=True)
        chlorides = st.number_input("Chlorides (g/dm³)", min_value=0.0, value=0.05, step=0.001,
                                   help="Sodium chloride concentration")
        sulphates = st.number_input("Sulphates (g/dm³)", min_value=0.0, value=0.5, step=0.01,
                                   help="Potassium sulphate level")
        alcohol = st.number_input("Alcohol (% vol)", min_value=0.0, value=11.0, step=0.1,
                                 help="Ethanol content")
        
        # Physical Properties
        st.markdown('<div class="feature-header">Physical Properties</div>', unsafe_allow_html=True)
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            density = st.number_input("Density (g/cm³)", min_value=0.0, value=0.995, step=0.0001,
                                     help="Mass per unit volume")
        with dcol2:
            pH = st.number_input("pH", min_value=0.0, value=3.3, step=0.01,
                                help="Acidity/basicity measurement")
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close input-section container

    # Form submission
    submitted = st.form_submit_button("ANALYZE WINE COMPOSITION", type="primary")

# Prediction logic with refined presentation
if submitted:
    with st.spinner("🔬 Analyzing wine composition with precision..."):
        try:
            input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                    chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                                    density, pH, sulphates, alcohol]])

            # Apply imputers
            median_input = input_data[:, [2, 8]]
            median_transformed = imputer_median.transform(median_input)
            input_data[:, 2] = median_transformed[:, 0]  # citric_acid
            input_data[:, 8] = median_transformed[:, 1]  # pH
            
            mean_input = input_data[:, [7]]
            mean_transformed = imputer_mean.transform(mean_input)
            input_data[:, 7] = mean_transformed[:, 0]  # density

            prediction = model.predict(input_data)[0]
            confidence = model.predict_proba(input_data)[0][prediction]
            confidence_pct = round(float(confidence) * 100, 2)
            
            # Visual feedback
            if prediction == 1:
                st.balloons()
            
            # Results card with refined design
            card_class = "good-wine" if prediction == 1 else "bad-wine"
            
            st.markdown(f'<div class="prediction-card {card_class}">', unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown("""
                <div style="text-align:center; margin-bottom:25px;">
                    <h1 style="font-family: 'Playfair Display', serif; font-size: 2.6rem; 
                    letter-spacing: 1px; margin-bottom: 15px; color: #2e7d32;">🏆 PREMIUM QUALITY</h1>
                    <p style="font-size: 1.2rem; font-family: 'Lora', serif; color: #388e3c;">
                    Exceptional characteristics worthy of our reserve collection</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align:center; margin-bottom:25px;">
                    <h1 style="font-family: 'Playfair Display', serif; font-size: 2.6rem; 
                    letter-spacing: 1px; margin-bottom: 15px; color: #c62828;">🔍 REFINEMENT NEEDED</h1>
                    <p style="font-size: 1.2rem; font-family: 'Lora', serif; color: #e53935;">
                    Potential uncovered through careful analysis</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence meter
            st.markdown(f'<div class="confidence-meter">', unsafe_allow_html=True)
            st.markdown(f'<h3 style="font-family: \'Lora\', serif; text-align: center; margin-bottom: 20px;">Model Confidence: {confidence_pct}%</h3>', 
                        unsafe_allow_html=True)
            st.progress(float(confidence))
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Expert recommendations
            if prediction == 1:
                st.markdown("### 🏅 Vintner's Assessment")
                st.markdown("""
                <div class="wine-tip">
                This vintage demonstrates remarkable qualities:
                - Perfectly balanced acidity profile (pH: {:.2f})
                - Elegant tannin structure (Sulphates: {:.2f} g/dm³)
                - Complex aromatic bouquet (Alcohol: {:.1f}% vol)
                - Long, satisfying finish
                </div>
                """.format(pH, sulphates, alcohol), unsafe_allow_html=True)
                
                st.markdown("### 🥂 Cellaring Recommendations")
                st.markdown("""
                - **Optimal serving**: 16-18°C (60-65°F)  
                - **Decanting time**: 30-45 minutes  
                - **Peak maturity**: 3-5 years  
                - **Food pairings**: Lamb, truffle dishes, aged gouda  
                """)
            else:
                st.markdown("### 🔬 Improvement Strategy")
                st.markdown(f"""
                <div class="wine-tip">
                Consider these refinement opportunities:
                - Acidity balance (current pH: {pH:.2f}, target: 3.2-3.6)
                - Sulfur optimization (current Total SO₂: {total_sulfur_dioxide:.2f} mg/dm³)
                - Alcohol-sugar harmony (current Alcohol: {alcohol:.1f}% vol)
                - Oak integration (current Sulphates: {sulphates:.2f} g/dm³)
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("### 🧪 Technical Insights")
                st.markdown(f"""
                - Volatile acidity: {volatile_acidity:.2f} g/dm³ (target < 0.5)  
                - Citric acid: {citric_acid:.2f} g/dm³ (target 0.2-0.5)  
                - Consider extended maceration  
                - Evaluate malolactic fermentation duration  
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close prediction card
            
        except Exception as e:
            st.error(f"🔍 Analysis Exception: {str(e)}")

# Footer with refined branding
st.markdown("""
<footer>
    <p style="font-family: 'Playfair Display', serif; font-size: 1.2rem; letter-spacing: 1.5px; margin-bottom: 5px;">Sanborn boutique wine</p>
    <p>Premier Wine Quality Assurance System</p>
    <p style="margin-top: 20px; font-size: 0.9rem;">Model v2.1 | Trained on 1,599 expert-rated wines | 95% accuracy</p>
    <p style="font-size:0.85rem; margin-top:15px; opacity: 0.8;">© 2025 Davao City • Crafted with precision</p>
</footer>
""", unsafe_allow_html=True)