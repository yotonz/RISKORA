import base64
import random
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from chatbot import chatbot_response
from db import (
    add_user,
    delete_application,
    get_all_applications,
    get_all_users,
    get_application_stats,
    get_user_applications,
    init_db,
    insert_application,
    validate_login,
)
from model import get_feature_importances, predict
from utils.helpers import credit_score_band, dti_band, fmt_percent
from utils.rules import rule_override
from utils.scoring import calculate_dti, financial_score, score_label
from utils.validation import validate, validate_email, validate_phone

# ── Brand ─────────────────────────────────────────────────────────────────────
APP_NAME    = "RiskOra AI"
APP_SHORT   = "RiskOra"
APP_TAGLINE = "Illuminate Risk. Lend with Confidence."
APP_SUB     = "AI-Powered Credit Risk Intelligence Platform"

# ── Floating Chat Widget HTML (plain string — __USER__ replaced at runtime) ───
_CHAT_HTML = """
<div id="__rc__" data-u="__USER__" style="display:none;height:0;position:absolute"></div>
<script>
(function(){
  if(document.getElementById('ro-fab'))return;
  var el=document.getElementById('__rc__');
  var U=el?el.getAttribute('data-u'):'User';

  /* ── Inject CSS ── */
  var s=document.createElement('style');s.id='ro-chat-css';
  s.textContent=[
    '#ro-fab{position:fixed;bottom:28px;right:28px;width:60px;height:60px;border-radius:50%;',
    'background:linear-gradient(135deg,#9333ea,#00d4ff);border:none;cursor:pointer;z-index:99999;',
    'box-shadow:0 0 0 0 rgba(147,51,234,.5);animation:ro-pulse 2.4s ease-in-out infinite;',
    'display:flex;align-items:center;justify-content:center;font-size:26px;transition:transform .25s;}',
    '#ro-fab:hover{transform:scale(1.13);}',
    '@keyframes ro-pulse{0%,100%{box-shadow:0 0 0 0 rgba(147,51,234,.55),0 6px 28px rgba(0,0,0,.5)}',
    '50%{box-shadow:0 0 0 14px rgba(147,51,234,0),0 6px 28px rgba(0,0,0,.5)}}',
    '#ro-badge{position:absolute;top:-2px;right:-2px;width:18px;height:18px;border-radius:50%;',
    'background:#ef4444;border:2px solid #06010f;font-size:9px;font-weight:800;color:#fff;',
    'display:flex;align-items:center;justify-content:center;animation:ro-badge-p 1.6s ease-in-out infinite;}',
    '@keyframes ro-badge-p{0%,100%{transform:scale(1)}50%{transform:scale(1.2)}}',
    '#ro-panel{position:fixed;bottom:100px;right:28px;width:380px;max-height:600px;',
    'background:linear-gradient(160deg,rgba(20,5,40,.98),rgba(8,1,20,.98));',
    'border:1px solid rgba(147,51,234,.4);border-radius:24px;z-index:99998;',
    'display:none;flex-direction:column;overflow:hidden;',
    'box-shadow:0 0 0 1px rgba(0,212,255,.08),0 0 60px rgba(147,51,234,.3),0 30px 80px rgba(0,0,0,.8);}',
    '#ro-panel.open{display:flex;}',
    '#ro-head{padding:16px 18px;background:linear-gradient(135deg,rgba(147,51,234,.2),rgba(0,212,255,.08));',
    'border-bottom:1px solid rgba(147,51,234,.22);display:flex;align-items:center;gap:12px;flex-shrink:0;}',
    '#ro-head-icon{width:40px;height:40px;border-radius:14px;background:linear-gradient(135deg,#9333ea,#00d4ff);',
    'display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0;',
    'box-shadow:0 0 20px rgba(147,51,234,.5);}',
    '#ro-head-text{flex:1;}',
    '#ro-head-name{font-size:14px;font-weight:700;color:#e2e8f0;}',
    '#ro-head-status{font-size:11px;color:#22c55e;display:flex;align-items:center;gap:5px;margin-top:2px;}',
    '#ro-head-status::before{content:"";width:7px;height:7px;border-radius:50%;background:#22c55e;',
    'animation:ro-badge-p 1.5s infinite;}',
    '#ro-close{background:rgba(255,255,255,.08);border:none;border-radius:8px;color:#94a3b8;',
    'cursor:pointer;width:30px;height:30px;font-size:16px;display:flex;align-items:center;justify-content:center;transition:all .2s;}',
    '#ro-close:hover{background:rgba(239,68,68,.15);color:#ef4444;}',
    '#ro-msgs{flex:1;overflow-y:auto;padding:14px 14px;display:flex;flex-direction:column;gap:10px;}',
    '#ro-msgs::-webkit-scrollbar{width:4px;}',
    '#ro-msgs::-webkit-scrollbar-thumb{background:rgba(147,51,234,.4);border-radius:2px;}',
    '.ro-msg-bot{display:flex;gap:8px;align-items:flex-start;}',
    '.ro-msg-user{display:flex;gap:8px;align-items:flex-start;justify-content:flex-end;}',
    '.ro-avatar{width:28px;height:28px;border-radius:50%;display:flex;align-items:center;',
    'justify-content:center;font-size:13px;flex-shrink:0;}',
    '.ro-av-bot{background:linear-gradient(135deg,#6366f1,#9333ea);}',
    '.ro-av-usr{background:linear-gradient(135deg,#9333ea,#06b6d4);}',
    '.ro-bubble-bot{background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.18);',
    'border-radius:16px 16px 16px 4px;padding:10px 14px;font-size:13px;color:#cbd5e1;',
    'line-height:1.65;max-width:88%;}',
    '.ro-bubble-usr{background:linear-gradient(135deg,rgba(147,51,234,.22),rgba(147,51,234,.08));',
    'border:1px solid rgba(147,51,234,.35);border-radius:16px 16px 4px 16px;',
    'padding:10px 14px;font-size:13px;color:#e2e8f0;line-height:1.65;max-width:88%;}',
    '.ro-typing{display:flex;gap:4px;align-items:center;padding:10px 14px;}',
    '.ro-dot{width:7px;height:7px;border-radius:50%;background:#9333ea;',
    'animation:ro-bounce .8s ease-in-out infinite;}',
    '.ro-dot:nth-child(2){animation-delay:.15s}.ro-dot:nth-child(3){animation-delay:.3s}',
    '@keyframes ro-bounce{0%,80%,100%{transform:scale(0)}40%{transform:scale(1)}}',
    '#ro-chips{padding:8px 14px;display:flex;flex-wrap:wrap;gap:6px;border-top:1px solid rgba(147,51,234,.1);flex-shrink:0;}',
    '.ro-chip{background:rgba(147,51,234,.1);border:1px solid rgba(147,51,234,.3);',
    'border-radius:50px;padding:4px 12px;font-size:11px;color:#c084fc;cursor:pointer;transition:all .2s;}',
    '.ro-chip:hover{background:rgba(147,51,234,.22);border-color:rgba(147,51,234,.55);}',
    '#ro-input-row{display:flex;gap:8px;padding:12px 14px;border-top:1px solid rgba(147,51,234,.12);flex-shrink:0;}',
    '#ro-input{flex:1;background:rgba(147,51,234,.06);border:1px solid rgba(147,51,234,.22);',
    'border-radius:12px;padding:10px 14px;font-size:13px;color:#e2e8f0;outline:none;font-family:inherit;}',
    '#ro-input:focus{border-color:rgba(147,51,234,.55);box-shadow:0 0 14px rgba(147,51,234,.18);}',
    '#ro-input::placeholder{color:#334155;}',
    '#ro-send{background:linear-gradient(135deg,#9333ea,#00d4ff);border:none;border-radius:10px;',
    'width:40px;height:40px;cursor:pointer;font-size:18px;color:#fff;transition:transform .2s;}',
    '#ro-send:hover{transform:scale(1.1);}'
  ].join('');
  document.head.appendChild(s);

  /* ── Knowledge base ── */
  var KB=[
    {k:['hello','hi','hey','greet','good morning','good evening','howdy'],
     a:'Hello __USER__! 👋 Welcome to RiskOra AI. I\'m your intelligent credit risk assistant. I can help you understand loan assessments, credit scores, DTI ratios, EMI calculations, and much more. What would you like to know?'},
    {k:['what is riskora','about riskora','about this app','what does this app','what can you do','features','capabilities'],
     a:'🔮 RiskOra AI is an AI-powered Credit Risk Intelligence Platform. It helps financial institutions predict loan default risk using:\n\n• 🤖 Random Forest ML model (100 trees)\n• 📊 Rule-based override engine\n• 🧮 EMI & DTI financial analysis\n• 📈 Real-time dashboards & analytics\n• 🔒 PBKDF2-SHA256 bank-grade security\n\nIt provides instant HIGH/LOW risk verdicts with full explanations.'},
    {k:['apply loan','loan application','how to apply','submit application','loan form'],
     a:'📋 To apply for a loan assessment:\n\n1. Click "Apply Loan" in the sidebar\n2. Fill in: Full name, age, annual income, loan amount, credit score, and monthly EMI\n3. Optionally add: email, phone, loan purpose, notes\n4. Click "Assess Risk" to get instant AI verdict\n\nThe system runs both an ML model and rule engine to give you a comprehensive risk assessment.'},
    {k:['high risk','high-risk','rejected','deny','denied'],
     a:'🔴 HIGH RISK indicates the loan application carries elevated default probability. This can be triggered by:\n\n• Credit score below 550\n• DTI ratio above 60%\n• Loan amount more than 6× annual income\n• ML model prediction based on overall profile\n\nHIGH RISK doesn\'t mean the loan is impossible — it means additional scrutiny or collateral may be required.'},
    {k:['low risk','low-risk','approved','safe','good','eligible'],
     a:'🟢 LOW RISK means the applicant has a strong financial profile with low default probability. Key indicators:\n\n• Credit score ≥ 650\n• DTI ratio below 40%\n• Loan within 4× annual income\n• Healthy EMI-to-income ratio\n\nLOW RISK applications typically qualify for better interest rates and faster approval.'},
    {k:['credit score','cibil','cibil score','credit rating','credit history'],
     a:'📊 Credit Score Bands in RiskOra:\n\n• 🔴 Below 550 — Very Poor (HIGH RISK auto-trigger)\n• 🟠 550–649 — Poor (elevated risk)\n• 🟡 650–699 — Fair (borderline)\n• 🟢 700–749 — Good (low risk)\n• 💎 750–799 — Very Good (preferred)\n• ⭐ 800+ — Excellent (best rates)\n\nCIBIL scores in India range from 300 to 900. A score of 750+ is considered excellent.'},
    {k:['dti','debt to income','debt-to-income','dti ratio','income ratio'],
     a:'📐 DTI (Debt-to-Income) Ratio measures your monthly debt burden:\n\nFormula: DTI = (Monthly EMI × 12 / Annual Income) × 100\n\nBands:\n• ✅ Below 30% — Excellent\n• 🟡 30–40% — Good\n• 🟠 40–50% — Fair (caution)\n• 🔴 50–60% — High\n• 🚨 Above 60% — Very High (HIGH RISK trigger)\n\nA DTI above 60% automatically triggers HIGH RISK in RiskOra.'},
    {k:['emi','equated monthly','monthly payment','emi calculator','calculate emi'],
     a:'🧮 EMI (Equated Monthly Installment) Calculator:\n\nFormula: EMI = P × r × (1+r)^n / ((1+r)^n - 1)\nWhere: P=Principal, r=Monthly interest rate, n=Tenure in months\n\nUse the EMI Calculator page in RiskOra (sidebar → EMI Calculator) to:\n• Calculate exact monthly payments\n• See full amortization schedule\n• Compare different tenures\n• View total interest payable'},
    {k:['financial score','score','risk score','loan score','scoring'],
     a:'💯 Financial Score (0–100) combines multiple factors:\n\n• Credit score health (major weight)\n• DTI ratio performance\n• Loan-to-income ratio\n• Age-income factor\n\nScore bands:\n• 80–100: Excellent financial health 🟢\n• 60–79: Good profile 🟡\n• 40–59: Fair — needs improvement 🟠\n• Below 40: Weak profile 🔴\n\nThe score is used alongside ML predictions for final assessment.'},
    {k:['ai assistant','chatbot','ai chat','assistant','ai help'],
     a:'🔮 The AI Assistant page (sidebar → AI Assistant) provides:\n\n• Conversational risk analysis Q&A\n• Context-aware answers about your submitted data\n• Quick-access question chips\n• Persistent chat history per session\n• Natural language interpretation of risk factors\n\nI\'m the floating widget version — always available on every page! The full assistant has deeper context about your specific application data.'},
    {k:['dashboard','analytics','charts','statistics','graphs','trends'],
     a:'📊 The Dashboard page provides:\n\n• Risk distribution pie chart\n• Income vs Loan scatter plot with regression\n• Credit score distribution histogram\n• DTI trend analysis\n• Top applicants table\n• Downloadable CSV export\n\nAdmin users see all applications; regular users see only their own submissions.'},
    {k:['admin','admin panel','administration','manage users','user management'],
     a:'🛡️ The Admin Panel (admin accounts only) provides:\n\n• User creation and management\n• View all registered accounts\n• Access all loan applications\n• Delete applications\n• Income vs Loan regression chart\n• Full CSV data export\n\nRegular users cannot access the Admin Panel. Contact your administrator to upgrade account roles.'},
    {k:['login','sign in','signin','log in'],
     a:'🔑 To log in:\n\n1. Go to the login page (you\'re already logged in as __USER__!)\n2. Enter your username and password\n3. Click "Sign In"\n\nIf you\'ve forgotten your password, contact your administrator. Passwords are hashed with PBKDF2-SHA256 (100,000 iterations) and cannot be recovered.'},
    {k:['signup','sign up','register','create account','new account'],
     a:'📝 To create a new account:\n\n1. Click "Sign Up" tab on the login page\n2. Choose a unique username\n3. Set a password (min 6 characters)\n4. Select role: user or admin\n5. Click "Create Account"\n\nPassword strength is shown in real-time. Your credentials are encrypted before storage.'},
    {k:['logout','log out','sign out','exit','end session'],
     a:'⏻ To logout:\n\nClick the "Logout" button at the top of the sidebar. Your session will be cleared immediately and you\'ll be returned to the login page.\n\nAll your submitted applications are safely stored in the database even after logout.'},
    {k:['security','password','encryption','hashing','safe','privacy','data protection'],
     a:'🔒 RiskOra Security Architecture:\n\n• Passwords: PBKDF2-SHA256, 100,000 iterations, random salt per user\n• Database: SQLite with parameterized queries (SQL injection proof)\n• Session: Streamlit server-side session state (no client cookies)\n• No plain-text storage of any credentials\n• Role-based access control (user / admin)\n• Input validation on all fields'},
    {k:['random forest','ml model','machine learning','model','algorithm','ai model','predict'],
     a:'🤖 RiskOra uses a Random Forest Classifier:\n\n• 100 decision trees, max depth 8\n• Features: Age, Annual Income, Loan Amount, Credit Score, Monthly EMI\n• Trained on synthetic Indian lending dataset\n• Cross-validated for accuracy\n• Model cached in memory for fast inference\n• Rule engine overrides ML for edge cases (credit_score<550, DTI>60%, loan>6× income)'},
    {k:['loan to income','loan ratio','6x','income ratio','loan limit'],
     a:'📏 Loan-to-Income Rule:\n\nRiskOra applies a hard rule: if Loan Amount > 6× Annual Income, the result is automatically HIGH RISK.\n\nExample: Income ₹6,00,000/year → Max safe loan = ₹36,00,000\n\nThis is a standard banking guideline. Most lenders prefer loan amounts within 3–4× annual income for comfortable repayment.'},
    {k:['sbi','state bank','state bank of india'],
     a:'🏦 State Bank of India (SBI) Helpdesk:\n\n📞 Customer Care: 1800 11 2211 (Toll-free)\n📞 Alternate: 1800 425 3800\n📞 Missed Call Banking: 09223488888\n\nServices: Home loans, personal loans, education loans, car loans, credit cards, and all retail banking queries.\n\n🌐 www.sbi.co.in | 24×7 available'},
    {k:['hdfc','hdfc bank'],
     a:'🏦 HDFC Bank Helpdesk:\n\n📞 Customer Care: 1800 202 6161 (Toll-free)\n📞 Alternate: 1800 258 3838\n📞 NRI Services: +91 22 6160 6161\n\nServices: Home loans, personal loans, auto loans, business loans, credit cards.\n\n🌐 www.hdfcbank.com | 24×7 available'},
    {k:['icici','icici bank'],
     a:'🏦 ICICI Bank Helpdesk:\n\n📞 Customer Care: 1800 200 3344 (Toll-free)\n📞 Alternate: 1860 120 7777\n📞 iMobile: Available on app\n\nServices: Home loans, personal loans, vehicle loans, business loans, credit cards.\n\n🌐 www.icicibank.com | 24×7 available'},
    {k:['axis','axis bank'],
     a:'🏦 Axis Bank Helpdesk:\n\n📞 Customer Care: 1800 419 5959 (Toll-free)\n📞 Alternate: 1800 209 5577\n📞 Priority Banking: 1800 103 5577\n\nServices: Home loans, personal loans, car loans, gold loans, credit cards.\n\n🌐 www.axisbank.com | 24×7 available'},
    {k:['kotak','kotak mahindra','kotak bank'],
     a:'🏦 Kotak Mahindra Bank Helpdesk:\n\n📞 Customer Care: 1860 266 2666\n📞 Alternate: 1800 209 0000 (Toll-free)\n📞 WhatsApp: 93222 87777\n\nServices: Home loans, personal loans, car loans, business loans.\n\n🌐 www.kotak.com | 24×7 available'},
    {k:['rbi','reserve bank','reserve bank of india','banking regulator'],
     a:'🏛️ Reserve Bank of India (RBI):\n\n📞 Complaints Helpline: 14448\n📞 Banking Ombudsman: 1800 22 1911 (Toll-free)\n📧 cms.rbi.org.in (Online complaint portal)\n\nRBI regulates all banks and NBFCs in India. Contact them for:\n• Unresolved bank complaints\n• Fraud reporting\n• Banking policy queries\n\n🌐 www.rbi.org.in'},
    {k:['eligibility','am i eligible','qualify','qualify for loan','loan eligibility'],
     a:'✅ General Loan Eligibility Criteria:\n\n• Age: 21–65 years\n• Minimum credit score: 650+ (some lenders accept 550+)\n• DTI ratio: Below 50% preferred\n• Stable income source\n• Loan ≤ 4–6× annual income\n• No major defaults in credit history\n\nRiskOra\'s AI gives you an instant assessment based on these parameters. Use the "Apply Loan" section to check eligibility.'},
    {k:['loan types','types of loan','home loan','personal loan','car loan','education loan','gold loan','business loan'],
     a:'🏦 Common Loan Types in India:\n\n• 🏠 Home Loan — 8–9.5% p.a., up to 30 years\n• 💼 Personal Loan — 10–24% p.a., 1–5 years\n• 🚗 Car/Auto Loan — 7–12% p.a., 1–7 years\n• 🎓 Education Loan — 8–15% p.a., up to 15 years\n• 🥇 Gold Loan — 7–12% p.a., up to 3 years\n• 🏢 Business Loan — 11–21% p.a., 1–5 years\n\nRates vary by bank and applicant profile.'},
    {k:['interest rate','rate','roi','rate of interest','percent'],
     a:'💹 Typical Loan Interest Rates in India (2026):\n\n• Home Loan: 8.40%–9.85% p.a.\n• Personal Loan: 10.5%–24% p.a.\n• Car Loan: 7.25%–12% p.a.\n• Education Loan: 8%–15% p.a.\n• Business Loan: 11%–21% p.a.\n\nRates are linked to RBI Repo Rate (currently ~6.5%). Better credit score = lower interest rate. Use the EMI Calculator to estimate payments at different rates.'},
    {k:['contact','support','help','helpdesk','agent','human','talk to someone'],
     a:'📞 Need to speak with our support team?\n\nAll agents are currently assisting other customers. Here\'s how to reach us:\n\n📧 Email: support@riskora.ai\n📞 Helpline: 1800-RISKORA (1800-747-5672)\n🕐 Hours: Mon–Fri, 9 AM – 6 PM IST\n\nOr visit the 📞 Support page in the sidebar to:\n• Browse FAQs\n• Read help articles\n• Submit a support ticket\n\nWe typically respond within 2–4 business hours.'},
    {k:['thank','thanks','thank you','great','perfect','awesome','nice'],
     a:'You\'re welcome, __USER__! 😊 Happy to help. Is there anything else you\'d like to know about RiskOra, your loan assessment, credit scores, or anything else? I\'m here 24/7!'},
    {k:['bye','goodbye','see you','cya','take care'],
     a:'Goodbye __USER__! 👋 Have a great day. Come back anytime you need help with credit risk assessment or any financial queries. RiskOra is always here for you! 🔮'}
  ];

  function matchKB(q){
    var ql=q.toLowerCase();
    for(var i=0;i<KB.length;i++){
      var entry=KB[i];
      for(var j=0;j<entry.k.length;j++){
        if(ql.indexOf(entry.k[j])!==-1) return entry.a;
      }
    }
    return null;
  }

  /* ── Build DOM ── */
  var fab=document.createElement('button');
  fab.id='ro-fab';fab.innerHTML='<span style="position:relative">💬<span id="ro-badge">1</span></span>';
  document.body.appendChild(fab);

  var panel=document.createElement('div');
  panel.id='ro-panel';
  panel.innerHTML=[
    '<div id="ro-head">',
    '<div id="ro-head-icon">🔮</div>',
    '<div id="ro-head-text">',
    '<div id="ro-head-name">RiskOra Assistant</div>',
    '<div id="ro-head-status">Online · Ready to help</div>',
    '</div>',
    '<button id="ro-close">✕</button>',
    '</div>',
    '<div id="ro-msgs"></div>',
    '<div id="ro-chips">',
    '<span class="ro-chip">Credit Score</span>',
    '<span class="ro-chip">DTI Ratio</span>',
    '<span class="ro-chip">EMI Calculator</span>',
    '<span class="ro-chip">Loan Eligibility</span>',
    '<span class="ro-chip">SBI Helpline</span>',
    '<span class="ro-chip">Contact Support</span>',
    '</div>',
    '<div id="ro-input-row">',
    '<input id="ro-input" placeholder="Type your question…" autocomplete="off"/>',
    '<button id="ro-send">➤</button>',
    '</div>'
  ].join('');
  document.body.appendChild(panel);

  var msgs=document.getElementById('ro-msgs');
  var input=document.getElementById('ro-input');
  var badge=document.getElementById('ro-badge');
  var isOpen=false;

  function addMsg(text,isBot){
    var wrap=document.createElement('div');
    wrap.className=isBot?'ro-msg-bot':'ro-msg-user';
    var av=document.createElement('div');
    av.className='ro-avatar '+(isBot?'ro-av-bot':'ro-av-usr');
    av.textContent=isBot?'🔮':'👤';
    var bub=document.createElement('div');
    bub.className=isBot?'ro-bubble-bot':'ro-bubble-usr';
    bub.style.whiteSpace='pre-wrap';
    bub.textContent=text;
    if(isBot){wrap.appendChild(av);wrap.appendChild(bub);}
    else{wrap.appendChild(bub);wrap.appendChild(av);}
    msgs.appendChild(wrap);
    msgs.scrollTop=msgs.scrollHeight;
  }

  function typingIndicator(){
    var wrap=document.createElement('div');
    wrap.className='ro-msg-bot';wrap.id='ro-typing-wrap';
    var av=document.createElement('div');
    av.className='ro-avatar ro-av-bot';av.textContent='🔮';
    var bub=document.createElement('div');
    bub.className='ro-bubble-bot ro-typing';
    bub.innerHTML='<div class="ro-dot"></div><div class="ro-dot"></div><div class="ro-dot"></div>';
    wrap.appendChild(av);wrap.appendChild(bub);
    msgs.appendChild(wrap);
    msgs.scrollTop=msgs.scrollHeight;
    return wrap;
  }

  function respond(q){
    addMsg(q,false);
    var tw=typingIndicator();
    setTimeout(function(){
      msgs.removeChild(tw);
      var ans=matchKB(q);
      if(!ans) ans='⚠️ All agents are currently busy assisting other customers.\\n\\nFor immediate help:\\n📞 Helpline: 1800-RISKORA (1800-747-5672)\\n📧 Email: support@riskora.ai\\n🕐 Mon–Fri, 9 AM – 6 PM IST\\n\\nOr visit the 📞 Support page in the sidebar to submit a ticket — our team responds within 2–4 hours.';
      ans=ans.replace(/__USER__/g,U);
      addMsg(ans,true);
    },800+Math.random()*500);
  }

  function openPanel(){
    panel.classList.add('open');
    isOpen=true;
    badge.style.display='none';
    if(!msgs.children.length){
      setTimeout(function(){
        addMsg('Hello '+U+'! 👋 Welcome to RiskOra AI Support. I can help you with:\n\n• Loan assessments & risk scores\n• Credit score & DTI ratio queries\n• EMI calculations\n• Bank helpline numbers\n• App features & how-to guides\n\nWhat would you like to know?',true);
      },300);
    }
  }

  fab.addEventListener('click',function(){
    if(isOpen){panel.classList.remove('open');isOpen=false;}
    else{openPanel();}
  });
  document.getElementById('ro-close').addEventListener('click',function(){
    panel.classList.remove('open');isOpen=false;
  });
  document.getElementById('ro-send').addEventListener('click',function(){
    var v=input.value.trim();if(!v)return;
    input.value='';respond(v);
  });
  input.addEventListener('keydown',function(e){
    if(e.key==='Enter'){var v=input.value.trim();if(!v)return;input.value='';respond(v);}
  });
  document.querySelectorAll('.ro-chip').forEach(function(c){
    c.addEventListener('click',function(){respond(c.textContent);});
  });
})();
</script>
"""


def show_chat_widget():
    uname = st.session_state.get("username", "User")
    safe  = uname.replace('"','').replace('<','').replace('>','').replace("'","")
    st.html(_CHAT_HTML.replace("__USER__", safe), unsafe_allow_javascript=True)


# ── Boot ──────────────────────────────────────────────────────────────────────
init_db()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role      = None
    st.session_state.username  = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "auth_tab" not in st.session_state:
    st.session_state.auth_tab = "login"
if "page" not in st.session_state:
    st.session_state.page = "landing"

st.set_page_config(
    page_title=f"{APP_NAME} — Credit Risk Intelligence",
    layout="wide",
    page_icon="🔮",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK SVG BACKGROUND
# ══════════════════════════════════════════════════════════════════════════════
def _build_neural_svg() -> str:
    W, H = 1920, 1080
    rng  = random.Random(7)
    layers = [
        (70,  [190, 360, 530, 700, 870]),
        (240, [130, 270, 410, 550, 690, 830, 970]),
        (440, [100, 210, 320, 430, 540, 650, 760, 870, 980]),
        (680, [140, 255, 370, 485, 600, 715, 830, 945]),
        (920, [170, 295, 420, 545, 670, 795, 920]),
        (1140,[200, 335, 470, 605, 740, 875]),
        (1340,[240, 390, 540, 690, 840]),
        (1510,[290, 460, 630, 800]),
        (1660,[360, 550, 740]),
        (1800,[450, 660]),
    ]
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">']
    p.append('''<defs>
<linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
  <stop offset="0%"   stop-color="#04010f"/>
  <stop offset="45%"  stop-color="#080120"/>
  <stop offset="100%" stop-color="#04010f"/>
</linearGradient>
<radialGradient id="o1" cx="16%" cy="36%" r="50%">
  <stop offset="0%"   stop-color="#9333ea" stop-opacity=".28"/>
  <stop offset="100%" stop-color="#04010f" stop-opacity="0"/>
</radialGradient>
<radialGradient id="o2" cx="84%" cy="14%" r="44%">
  <stop offset="0%"   stop-color="#00d4ff" stop-opacity=".16"/>
  <stop offset="100%" stop-color="#04010f" stop-opacity="0"/>
</radialGradient>
<radialGradient id="o3" cx="55%" cy="86%" r="40%">
  <stop offset="0%"   stop-color="#ec4899" stop-opacity=".10"/>
  <stop offset="100%" stop-color="#04010f" stop-opacity="0"/>
</radialGradient>
<radialGradient id="o4" cx="50%" cy="50%" r="35%">
  <stop offset="0%"   stop-color="#6366f1" stop-opacity=".08"/>
  <stop offset="100%" stop-color="#04010f" stop-opacity="0"/>
</radialGradient>
<filter id="glow"><feGaussianBlur stdDeviation="3" result="b"/>
  <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
<filter id="glow-lg"><feGaussianBlur stdDeviation="6" result="b"/>
  <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs>''')
    p += [
        f'<rect width="{W}" height="{H}" fill="url(#bg)"/>',
        f'<rect width="{W}" height="{H}" fill="url(#o1)"/>',
        f'<rect width="{W}" height="{H}" fill="url(#o2)"/>',
        f'<rect width="{W}" height="{H}" fill="url(#o3)"/>',
        f'<rect width="{W}" height="{H}" fill="url(#o4)"/>',
    ]
    p.append('<g fill="#9333ea" fill-opacity=".02">')
    for x in range(0, W, 55):
        for y in range(0, H, 55):
            p.append(f'<circle cx="{x}" cy="{y}" r="1"/>')
    p.append('</g>')
    p.append('<g stroke="#9333ea" stroke-opacity=".05" stroke-width=".55">')
    for i in range(len(layers) - 1):
        x1, ys1 = layers[i]; x2, ys2 = layers[i + 1]
        for y1 in ys1:
            for y2 in ys2:
                if abs(y2 - y1) < 310:
                    p.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>')
    p.append('</g>')
    p.append('<g stroke="#00d4ff" stroke-opacity=".28" stroke-width="1.5">')
    for i in range(len(layers) - 1):
        x1, ys1 = layers[i]; x2, ys2 = layers[i + 1]
        m1 = ys1[len(ys1) // 2]; m2 = ys2[len(ys2) // 2]
        p.append(f'<line x1="{x1}" y1="{m1}" x2="{x2}" y2="{m2}"/>')
    p.append('</g>')
    p.append('<g stroke="#9333ea" stroke-opacity=".2" stroke-width="1.1">')
    for i in range(len(layers) - 1):
        x1, ys1 = layers[i]; x2, ys2 = layers[i + 1]
        if len(ys1) > 1 and len(ys2) > 1:
            p.append(f'<line x1="{x1}" y1="{ys1[1]}" x2="{x2}" y2="{ys2[1]}"/>')
    p.append('</g>')
    p.append('<g filter="url(#glow)">')
    all_nodes = [(x, y, li) for li, (x, ys) in enumerate(layers) for y in ys]
    for idx, (nx, ny, li) in enumerate(all_nodes):
        color = "#00d4ff" if li % 3 != 1 else "#9333ea"
        r     = 4.5 if idx % 7 == 0 else (3.5 if idx % 3 == 0 else 2.8)
        op    = min(0.92, 0.38 + (idx % 7) * 0.08)
        p.append(f'<circle cx="{nx}" cy="{ny}" r="{r}" fill="{color}" fill-opacity="{op:.2f}"/>')
    p.append('</g>')
    p.append('<g filter="url(#glow-lg)" fill="none" stroke-width="1.2">')
    for li in [2, 4, 6, 8]:
        if li < len(layers):
            x, ys = layers[li]
            mid   = ys[len(ys) // 2]
            p.append(f'<circle cx="{x}" cy="{mid}" r="10" stroke="#9333ea" stroke-opacity=".5"/>')
            p.append(f'<circle cx="{x}" cy="{mid}" r="20" stroke="#00d4ff" stroke-opacity=".12"/>')
    p.append('</g>')
    p.append('<g>')
    for _ in range(60):
        px = rng.randint(40, W - 40); py = rng.randint(30, H - 30)
        pr = round(rng.uniform(0.7, 2.4), 1); po = round(rng.uniform(0.06, 0.25), 2)
        pc = rng.choice(["#00d4ff", "#9333ea", "#ec4899", "#6366f1"])
        p.append(f'<circle cx="{px}" cy="{py}" r="{pr}" fill="{pc}" fill-opacity="{po}"/>')
    p.append('</g>')
    p.append('</svg>')
    return '\n'.join(p)


@st.cache_data(show_spinner=False)
def _neural_bg_url() -> str:
    svg     = _build_neural_svg()
    encoded = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
    return f"url(\"data:image/svg+xml;base64,{encoded}\")"


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
def _inject_css() -> None:
    neural_url = _neural_bg_url()
    logged_in  = st.session_state.logged_in

    if logged_in:
        bg = """background-color:#06010f;
    background-image:
        radial-gradient(ellipse 70% 50% at 10% 30%,rgba(147,51,234,.09) 0%,transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 10%,rgba(0,212,255,.07) 0%,transparent 60%),
        radial-gradient(ellipse 50% 60% at 55% 80%,rgba(236,72,153,.05) 0%,transparent 60%),
        linear-gradient(rgba(147,51,234,.015) 1px,transparent 1px),
        linear-gradient(90deg,rgba(0,212,255,.015) 1px,transparent 1px);
    background-size:cover,cover,cover,46px 46px,46px 46px;"""
    else:
        bg = f"""background-image:{neural_url};
    background-size:cover;background-position:center;background-attachment:fixed;"""

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Orbitron:wght@600;900&display=swap');

/* ── Animations ── */
@keyframes float      {{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-10px)}}}}
@keyframes glow-pulse {{0%,100%{{box-shadow:0 0 20px rgba(147,51,234,.4),0 0 40px rgba(0,212,255,.15)}}
                        50%{{box-shadow:0 0 50px rgba(147,51,234,.7),0 0 100px rgba(0,212,255,.3)}}}}
@keyframes fade-up    {{from{{opacity:0;transform:translateY(22px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes pulse-ring {{0%{{transform:scale(1);opacity:.6}}100%{{transform:scale(1.6);opacity:0}}}}
@keyframes badge-pulse{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.82;transform:scale(1.025)}}}}
@keyframes scan       {{0%{{transform:translateY(-100%)}}100%{{transform:translateY(600px)}}}}
@keyframes shimmer    {{0%{{background-position:-400% 0}}100%{{background-position:400% 0}}}}
@keyframes spin-slow  {{0%{{transform:rotate(0deg)}}100%{{transform:rotate(360deg)}}}}
@keyframes gradient-x {{0%,100%{{background-position:0% 50%}}50%{{background-position:100% 50%}}}}

/* ── Root ── */
html,body,[data-testid="stAppViewContainer"]{{
    font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;color:#e2e8f0;
}}
[data-testid="stAppViewContainer"]{{ {bg} }}
[data-testid="stHeader"]  {{background:transparent!important;}}
[data-testid="stToolbar"] {{display:none;}}
.block-container          {{padding-top:2rem;}}

/* ── Sidebar ── */
[data-testid="stSidebar"]{{
    background:linear-gradient(180deg,rgba(10,2,25,.98),rgba(6,1,15,.98))!important;
    border-right:1px solid rgba(147,51,234,.18)!important;
    box-shadow:4px 0 50px rgba(147,51,234,.12)!important;
}}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span{{color:#94a3b8!important;font-size:13px!important;}}
[data-testid="stSidebar"] .stSelectbox > div > div{{
    background:rgba(147,51,234,.06)!important;
    border:1px solid rgba(147,51,234,.22)!important;
    border-radius:10px!important;color:#e2e8f0!important;
}}

/* ── Buttons ── */
.stButton > button{{
    background:linear-gradient(135deg,#7c3aed,#0891b2)!important;
    color:white!important;border:none!important;
    border-radius:12px!important;height:46px!important;
    font-size:14px!important;font-weight:600!important;letter-spacing:.5px!important;
    box-shadow:0 0 22px rgba(147,51,234,.35),0 4px 15px rgba(0,0,0,.5)!important;
    transition:all .25s ease!important;width:100%!important;
}}
.stButton > button:hover{{
    box-shadow:0 0 42px rgba(147,51,234,.6),0 6px 22px rgba(0,0,0,.6)!important;
    transform:translateY(-2px)!important;
    background:linear-gradient(135deg,#9333ea,#06b6d4)!important;
}}

/* ── Auth tab toggle buttons ── */
.auth-tab-active > .stButton > button{{
    background:linear-gradient(135deg,#9333ea,#00d4ff)!important;
    box-shadow:0 0 28px rgba(147,51,234,.5),0 4px 15px rgba(0,0,0,.4)!important;
    font-weight:800!important;
}}
.auth-tab-inactive > .stButton > button{{
    background:rgba(255,255,255,.04)!important;
    border:1px solid rgba(255,255,255,.1)!important;
    color:#64748b!important;box-shadow:none!important;
}}
.auth-tab-inactive > .stButton > button:hover{{
    background:rgba(147,51,234,.12)!important;
    box-shadow:0 0 18px rgba(147,51,234,.25)!important;
    color:#e2e8f0!important;
}}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea{{
    background:rgba(147,51,234,.06)!important;color:#e2e8f0!important;
    border:1px solid rgba(147,51,234,.22)!important;border-radius:12px!important;font-size:14px!important;
}}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus{{
    border-color:rgba(147,51,234,.6)!important;
    box-shadow:0 0 18px rgba(147,51,234,.22)!important;
}}
.stSelectbox > div > div{{
    background:rgba(147,51,234,.06)!important;
    border:1px solid rgba(147,51,234,.22)!important;border-radius:12px!important;color:#e2e8f0!important;
}}
label{{color:#94a3b8!important;font-size:13px!important;font-weight:500!important;}}

/* ── Metrics ── */
[data-testid="metric-container"]{{
    background:rgba(147,51,234,.06);border:1px solid rgba(147,51,234,.16);
    border-radius:16px;padding:18px 20px;
}}
[data-testid="stMetricValue"]{{color:#e2e8f0!important;font-weight:700!important;}}
[data-testid="stMetricLabel"]{{color:#64748b!important;font-size:12px!important;}}

/* ── DataFrames ── */
[data-testid="stDataFrame"]{{
    border:1px solid rgba(147,51,234,.14)!important;border-radius:14px!important;overflow:hidden!important;
}}

/* ── Tabs ── */
[data-baseweb="tab-list"]{{
    background:rgba(147,51,234,.06)!important;
    border:1px solid rgba(147,51,234,.16)!important;
    border-radius:14px!important;padding:5px!important;gap:4px!important;
}}
[data-baseweb="tab"]{{
    color:#475569!important;font-weight:600!important;
    border-radius:10px!important;font-size:14px!important;padding:8px 22px!important;
    border:1px solid transparent!important;
}}
[aria-selected="true"][data-baseweb="tab"]{{
    background:linear-gradient(135deg,rgba(147,51,234,.28),rgba(0,212,255,.12))!important;
    color:#c084fc!important;border-color:rgba(147,51,234,.4)!important;
}}
[data-baseweb="tab-highlight"]{{display:none!important;}}
[data-testid="stTabContent"]{{padding-top:16px!important;}}

/* ── Alerts ── */
.stAlert,[data-baseweb="notification"]{{border-radius:14px!important;}}
::-webkit-scrollbar{{width:5px;}}
::-webkit-scrollbar-track{{background:rgba(255,255,255,.02);}}
::-webkit-scrollbar-thumb{{background:rgba(147,51,234,.4);border-radius:3px;}}

/* ══════════════════════════════════
   AUTH PAGE
══════════════════════════════════ */

/* Left hero */
.auth-hero{{padding:10px 0;animation:fade-up .6s ease both;}}

.brand-wordmark{{
    display:flex;align-items:center;gap:16px;margin-bottom:8px;
}}
.brand-icon-wrap{{
    width:60px;height:60px;border-radius:18px;
    background:linear-gradient(135deg,#9333ea,#00d4ff);
    display:flex;align-items:center;justify-content:center;font-size:28px;
    box-shadow:0 0 30px rgba(147,51,234,.55),0 0 60px rgba(0,212,255,.18);
    animation:glow-pulse 3s ease-in-out infinite;flex-shrink:0;
}}
.brand-name-text{{
    font-family:'Orbitron','Inter',sans-serif;font-size:38px;font-weight:900;letter-spacing:2px;
    background:linear-gradient(135deg,#ffffff 0%,#c084fc 40%,#00d4ff 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    line-height:1;
}}
.brand-tag{{
    font-family:'Orbitron',sans-serif;font-size:9px;font-weight:700;
    letter-spacing:4px;color:#9333ea;text-transform:uppercase;margin-top:4px;
}}
.hero-tagline{{
    font-size:28px;font-weight:800;line-height:1.25;margin-bottom:10px;
    background:linear-gradient(135deg,#e2e8f0,#c084fc,#00d4ff);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}}
.hero-desc{{font-size:14px;color:#64748b;line-height:1.85;margin-bottom:28px;max-width:430px;}}

/* Feature grid */
.feat-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:26px;}}
.feat-item{{
    border-radius:14px;padding:14px 16px;transition:all .25s;
    border:1px solid;
}}
.feat-item:hover{{transform:translateY(-2px);}}
.feat-c{{color:#00d4ff;  background:rgba(0,212,255,.06);  border-color:rgba(0,212,255,.18);}}
.feat-p{{color:#c084fc;  background:rgba(147,51,234,.07); border-color:rgba(147,51,234,.22);}}
.feat-t{{color:#2dd4bf;  background:rgba(20,184,166,.06); border-color:rgba(20,184,166,.18);}}
.feat-i{{color:#818cf8;  background:rgba(99,102,241,.07); border-color:rgba(99,102,241,.2);}}
.feat-a{{color:#fbbf24;  background:rgba(245,158,11,.06); border-color:rgba(245,158,11,.18);}}
.feat-e{{color:#34d399;  background:rgba(16,185,129,.06); border-color:rgba(16,185,129,.18);}}
.feat-item-icon{{font-size:19px;margin-bottom:7px;}}
.feat-item-title{{font-size:13px;font-weight:700;color:#e2e8f0;margin-bottom:3px;}}
.feat-item-desc{{font-size:11px;color:#475569;line-height:1.5;}}

/* How it works */
.how-label{{font-size:10px;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:#1e293b;margin-bottom:14px;}}
.how-steps{{display:flex;align-items:flex-start;margin-bottom:24px;}}
.how-step{{flex:1;text-align:center;}}
.step-num-c{{width:38px;height:38px;border-radius:50%;margin:0 auto 8px;display:flex;align-items:center;justify-content:center;font-size:14px;font-weight:900;}}
.step-num-1{{background:linear-gradient(135deg,rgba(99,102,241,.25),rgba(99,102,241,.08));border:1.5px solid rgba(99,102,241,.5);color:#818cf8;box-shadow:0 0 14px rgba(99,102,241,.25);}}
.step-num-2{{background:linear-gradient(135deg,rgba(147,51,234,.25),rgba(147,51,234,.08));border:1.5px solid rgba(147,51,234,.5);color:#c084fc;box-shadow:0 0 14px rgba(147,51,234,.25);}}
.step-num-3{{background:linear-gradient(135deg,rgba(0,212,255,.2),rgba(0,212,255,.05));border:1.5px solid rgba(0,212,255,.45);color:#00d4ff;box-shadow:0 0 14px rgba(0,212,255,.2);}}
.how-step-title{{font-size:12px;font-weight:700;color:#e2e8f0;margin-bottom:3px;}}
.how-step-desc{{font-size:11px;color:#475569;line-height:1.5;}}
.how-arrow{{flex:none;width:32px;display:flex;align-items:flex-start;justify-content:center;padding-top:19px;font-size:16px;color:rgba(147,51,234,.4);}}

/* Preview card */
.preview-card{{
    background:linear-gradient(135deg,rgba(10,2,25,.95),rgba(6,1,15,.9));
    border-radius:16px;padding:16px 20px;margin-bottom:22px;
    border:1px solid;border-image:linear-gradient(135deg,rgba(147,51,234,.4),rgba(0,212,255,.3),rgba(236,72,153,.2)) 1;
    position:relative;overflow:hidden;
}}
.preview-card::before{{
    content:'SAMPLE OUTPUT';position:absolute;top:12px;right:14px;
    font-size:9px;letter-spacing:2px;color:#1e3a5f;font-weight:700;
}}
.preview-name-row{{display:flex;align-items:center;gap:10px;margin-bottom:12px;}}
.preview-name{{font-size:14px;font-weight:700;color:#e2e8f0;}}
.p-low{{background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.35);border-radius:50px;padding:2px 12px;font-size:11px;font-weight:700;color:#22c55e;}}
.preview-stats{{display:flex;gap:18px;}}
.ps-val{{font-size:17px;font-weight:800;}}
.ps-lbl{{font-size:10px;color:#475569;margin-top:2px;}}

/* Trust badges */
.trust-row{{display:flex;gap:7px;flex-wrap:wrap;}}
.tb-c{{background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.22);border-radius:50px;padding:5px 13px;font-size:11px;color:#00d4ff;}}
.tb-p{{background:rgba(147,51,234,.1);border:1px solid rgba(147,51,234,.28);border-radius:50px;padding:5px 13px;font-size:11px;color:#c084fc;}}
.tb-g{{background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.22);border-radius:50px;padding:5px 13px;font-size:11px;color:#22c55e;}}
.tb-a{{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.2);border-radius:50px;padding:5px 13px;font-size:11px;color:#fbbf24;}}

/* Auth card (right panel) */
.auth-card{{
    background:linear-gradient(160deg,rgba(20,5,40,.95),rgba(8,1,20,.95));
    backdrop-filter:blur(40px);
    border-radius:28px;padding:32px 36px;
    border:1px solid rgba(147,51,234,.35);
    box-shadow:
        0 0 0 1px rgba(0,212,255,.08),
        0 0 70px rgba(147,51,234,.22),
        0 30px 80px rgba(0,0,0,.75),
        inset 0 1px 0 rgba(255,255,255,.06);
    position:relative;overflow:hidden;
    animation:fade-up .5s ease both;
}}
.auth-card::before{{
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent 0%,#9333ea 30%,#00d4ff 60%,#ec4899 80%,transparent 100%);
    animation:shimmer 3s linear infinite;background-size:200% 100%;
}}
.auth-brand-row{{
    display:flex;align-items:center;justify-content:center;gap:12px;margin-bottom:20px;
}}
.auth-icon{{
    width:44px;height:44px;border-radius:14px;
    background:linear-gradient(135deg,#9333ea,#00d4ff);
    display:flex;align-items:center;justify-content:center;font-size:20px;
    box-shadow:0 0 24px rgba(147,51,234,.5),0 0 48px rgba(0,212,255,.15);
}}
.auth-brand-name{{
    font-family:'Orbitron',sans-serif;font-size:22px;font-weight:900;
    background:linear-gradient(135deg,#fff,#c084fc,#00d4ff);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    letter-spacing:2px;
}}
.auth-divider{{height:1px;background:linear-gradient(90deg,transparent,rgba(147,51,234,.3),rgba(0,212,255,.2),transparent);margin:16px 0;}}
.auth-form-title{{font-size:20px;font-weight:800;color:#f1f5f9;margin-bottom:4px;}}
.auth-form-sub{{font-size:13px;color:#475569;margin-bottom:20px;}}
.auth-secure-note{{
    text-align:center;margin-top:18px;padding-top:14px;
    border-top:1px solid rgba(255,255,255,.05);
    font-size:11px;color:#334155;
}}

/* Password strength */
.pwd-bar-wrap{{margin-top:4px;display:flex;gap:5px;}}
.pwd-seg{{flex:1;height:4px;border-radius:2px;background:rgba(255,255,255,.08);transition:all .3s;}}

/* ══════════════════════════════════
   APP COMPONENTS
══════════════════════════════════ */
.gc{{
    background:rgba(255,255,255,.025);border:1px solid rgba(147,51,234,.14);
    border-radius:20px;padding:26px 28px;margin-bottom:16px;
    backdrop-filter:blur(20px);animation:fade-up .5s ease both;
    transition:border-color .25s,box-shadow .25s,transform .25s;
}}
.gc:hover{{
    border-color:rgba(147,51,234,.36);
    box-shadow:0 0 32px rgba(147,51,234,.1),0 14px 44px rgba(0,0,0,.42);
    transform:translateY(-2px);
}}
.glow-card{{
    background:linear-gradient(135deg,rgba(147,51,234,.08),rgba(0,212,255,.06));
    border:1px solid rgba(147,51,234,.28);border-radius:20px;padding:26px 28px;margin-bottom:16px;
    animation:glow-pulse 3s ease-in-out infinite;
}}
.page-title{{
    font-family:'Orbitron','Inter',sans-serif;font-size:34px;font-weight:900;
    background:linear-gradient(135deg,#fff 0%,#c084fc 50%,#00d4ff 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    margin-bottom:4px;
}}
.page-sub{{font-size:14px;color:#475569;margin-bottom:24px;}}
.slabel{{font-size:10px;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:#9333ea;margin-bottom:6px;}}
.htitle{{
    font-family:'Orbitron','Inter',sans-serif;font-size:44px;font-weight:900;
    line-height:1.1;margin-bottom:14px;
    background:linear-gradient(135deg,#ffffff 0%,#c084fc 45%,#00d4ff 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
}}
.ai-chip{{
    display:inline-flex;align-items:center;gap:7px;
    background:linear-gradient(135deg,rgba(147,51,234,.2),rgba(0,212,255,.06));
    border:1px solid rgba(147,51,234,.45);border-radius:50px;
    padding:6px 18px;font-size:11px;font-weight:700;letter-spacing:2px;
    color:#c084fc;margin-bottom:22px;text-transform:uppercase;animation:glow-pulse 3s ease infinite;
}}
.stat-row{{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px;}}
.stat-c{{
    background:rgba(255,255,255,.03);border:1px solid rgba(147,51,234,.16);
    border-radius:16px;padding:18px 22px;flex:1;min-width:130px;
    position:relative;overflow:hidden;transition:all .25s;
}}
.stat-c::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,#9333ea,#00d4ff);}}
.stat-c:hover{{border-color:rgba(147,51,234,.42);box-shadow:0 0 24px rgba(147,51,234,.12);}}
.stat-val{{font-size:28px;font-weight:800;color:#fff;line-height:1;}}
.stat-lbl{{font-size:11px;color:#64748b;margin-top:4px;}}
.stat-up{{font-size:11px;color:#22c55e;font-weight:600;margin-top:4px;}}
.stat-dn{{font-size:11px;color:#ef4444;font-weight:600;margin-top:4px;}}
.float-card{{
    background:rgba(10,2,25,.92);border:1px solid rgba(147,51,234,.28);
    border-radius:16px;padding:14px 18px;margin-bottom:12px;backdrop-filter:blur(20px);
    animation:fade-up .6s ease both;transition:all .25s;
}}
.float-card:hover{{border-color:rgba(147,51,234,.55);box-shadow:0 0 22px rgba(147,51,234,.16);}}
.fc-lbl{{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#475569;}}
.fc-val{{font-size:22px;font-weight:800;color:#fff;margin:4px 0 2px;}}
.fc-sub{{font-size:12px;color:#9333ea;}}
.ai-orb{{
    width:130px;height:130px;border-radius:50%;
    background:radial-gradient(circle at 35% 35%,rgba(147,51,234,.45),rgba(0,212,255,.2),rgba(0,0,0,.7));
    border:2px solid rgba(147,51,234,.6);
    display:flex;align-items:center;justify-content:center;font-size:52px;
    box-shadow:0 0 55px rgba(147,51,234,.55),0 0 110px rgba(0,212,255,.2),
               inset 0 0 30px rgba(147,51,234,.14);
    animation:float 4s ease-in-out infinite,glow-pulse 3s ease-in-out infinite;
    margin:0 auto;
}}
.feat-card{{
    background:rgba(255,255,255,.02);border:1px solid rgba(147,51,234,.12);
    border-radius:22px;padding:28px 24px;height:100%;
    position:relative;overflow:hidden;transition:all .3s;animation:fade-up .7s ease both;
}}
.feat-card:hover{{
    border-color:rgba(147,51,234,.4);
    box-shadow:0 22px 64px rgba(0,0,0,.52),0 0 32px rgba(147,51,234,.1);
    transform:translateY(-4px);
}}
.feat-icon{{width:52px;height:52px;border-radius:14px;display:flex;align-items:center;justify-content:center;font-size:22px;margin-bottom:16px;}}
.feat-title{{font-size:18px;font-weight:700;color:#fff;margin-bottom:10px;}}
.feat-desc{{font-size:14px;color:#64748b;line-height:1.65;}}
.rb-high{{
    display:inline-block;padding:10px 32px;border-radius:50px;
    font-size:20px;font-weight:800;letter-spacing:2px;
    background:linear-gradient(135deg,rgba(239,68,68,.18),rgba(239,68,68,.04));
    border:2px solid #ef4444;color:#ef4444;
    box-shadow:0 0 26px rgba(239,68,68,.38),inset 0 0 22px rgba(239,68,68,.05);
    animation:badge-pulse 2s ease-in-out infinite;
}}
.rb-low{{
    display:inline-block;padding:10px 32px;border-radius:50px;
    font-size:20px;font-weight:800;letter-spacing:2px;
    background:linear-gradient(135deg,rgba(34,197,94,.18),rgba(34,197,94,.04));
    border:2px solid #22c55e;color:#22c55e;
    box-shadow:0 0 26px rgba(34,197,94,.38),inset 0 0 22px rgba(34,197,94,.05);
    animation:badge-pulse 2s ease-in-out infinite;
}}
.mrow{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0 8px;}}
.mbox{{
    background:rgba(147,51,234,.05);border:1px solid rgba(147,51,234,.14);
    border-radius:14px;padding:14px 18px;flex:1;min-width:120px;
}}
.mbox-lbl{{font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:1px;}}
.mbox-val{{font-size:22px;font-weight:700;color:#fff;margin-top:4px;}}
.mbox-sub{{font-size:12px;color:#94a3b8;margin-top:2px;}}
.sb-logo{{font-family:'Orbitron',sans-serif;font-size:15px;font-weight:900;
    background:linear-gradient(135deg,#c084fc,#00d4ff);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    letter-spacing:2px;}}
.sb-logo-sub{{font-size:9px;letter-spacing:3px;color:#1e293b;text-transform:uppercase;}}
.sb-user{{background:rgba(147,51,234,.08);border:1px solid rgba(147,51,234,.22);border-radius:14px;padding:12px 14px;margin:12px 0;}}
.sb-username{{font-size:14px;font-weight:700;color:#e2e8f0;}}
.sb-role-admin{{display:inline-block;margin-top:4px;padding:2px 10px;border-radius:50px;font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;background:rgba(147,51,234,.2);border:1px solid rgba(147,51,234,.5);color:#c084fc;}}
.sb-role-user{{display:inline-block;margin-top:4px;padding:2px 10px;border-radius:50px;font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.4);color:#38bdf8;}}
.sb-nav-label{{font-size:9px;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:#1e293b;padding:0 4px;margin:14px 0 6px;}}
.session-card{{background:linear-gradient(135deg,rgba(147,51,234,.07),rgba(0,212,255,.04));border:1px solid rgba(147,51,234,.18);border-radius:20px;padding:24px 28px;margin-top:8px;}}
.chat-user-wrap{{display:flex;justify-content:flex-end;margin:8px 0;}}
.chat-bot-wrap {{display:flex;justify-content:flex-start;margin:8px 0;}}
.chat-user{{background:linear-gradient(135deg,rgba(147,51,234,.22),rgba(147,51,234,.08));border:1px solid rgba(147,51,234,.35);border-radius:18px 18px 4px 18px;padding:12px 18px;max-width:75%;font-size:14px;color:#e2e8f0;line-height:1.6;}}
.chat-bot{{background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.18);border-radius:18px 18px 18px 4px;padding:14px 18px;max-width:80%;font-size:14px;color:#cbd5e1;line-height:1.75;white-space:pre-wrap;}}
.chat-avatar-u{{width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,#9333ea,#06b6d4);display:flex;align-items:center;justify-content:center;font-size:14px;margin-left:10px;flex-shrink:0;}}
.chat-avatar-b{{width:32px;height:32px;border-radius:50%;background:linear-gradient(135deg,#6366f1,#9333ea);display:flex;align-items:center;justify-content:center;font-size:14px;margin-right:10px;flex-shrink:0;align-self:flex-start;}}
.emi-card{{background:linear-gradient(135deg,rgba(147,51,234,.08),rgba(0,212,255,.06));border:1px solid rgba(147,51,234,.24);border-radius:20px;padding:24px;}}
.divider{{height:1px;background:linear-gradient(90deg,transparent,rgba(147,51,234,.3),rgba(0,212,255,.2),transparent);margin:24px 0;}}

/* ─── Support page ─── */
.sup-hero{{background:linear-gradient(135deg,rgba(147,51,234,.09),rgba(0,212,255,.06));border:1px solid rgba(147,51,234,.22);border-radius:24px;padding:32px 36px;margin-bottom:28px;position:relative;overflow:hidden;}}
.sup-hero::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#9333ea,#00d4ff,transparent);}}
.sup-kpi-row{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:28px;}}
.sup-kpi{{background:rgba(255,255,255,.025);border:1px solid rgba(147,51,234,.16);border-radius:16px;padding:18px 22px;flex:1;min-width:130px;text-align:center;}}
.sup-kpi-val{{font-size:26px;font-weight:800;color:#fff;line-height:1;}}
.sup-kpi-lbl{{font-size:11px;color:#64748b;margin-top:4px;}}
.bank-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:16px 0;}}
.bank-card{{background:rgba(255,255,255,.025);border:1px solid rgba(147,51,234,.16);border-radius:16px;padding:18px 16px;transition:all .25s;}}
.bank-card:hover{{border-color:rgba(147,51,234,.4);box-shadow:0 0 24px rgba(147,51,234,.1);transform:translateY(-2px);}}
.bank-name{{font-size:13px;font-weight:700;color:#e2e8f0;margin-bottom:6px;}}
.bank-num{{font-size:14px;font-weight:800;color:#00d4ff;letter-spacing:.5px;margin-bottom:4px;}}
.bank-sub{{font-size:11px;color:#475569;}}
.art-card{{background:rgba(255,255,255,.02);border:1px solid rgba(147,51,234,.12);border-radius:16px;padding:20px 22px;margin-bottom:10px;display:flex;gap:16px;align-items:flex-start;transition:all .25s;cursor:default;}}
.art-card:hover{{border-color:rgba(147,51,234,.35);background:rgba(147,51,234,.04);}}
.art-icon{{font-size:28px;flex-shrink:0;margin-top:2px;}}
.art-title{{font-size:14px;font-weight:700;color:#e2e8f0;margin-bottom:4px;}}
.art-desc{{font-size:13px;color:#64748b;line-height:1.6;}}
.art-tag{{display:inline-block;margin-top:8px;padding:2px 10px;border-radius:50px;font-size:10px;font-weight:700;letter-spacing:1px;}}
.art-tag-c{{background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.3);color:#00d4ff;}}
.art-tag-p{{background:rgba(147,51,234,.12);border:1px solid rgba(147,51,234,.35);color:#c084fc;}}
.art-tag-g{{background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.28);color:#22c55e;}}

/* ══ LANDING PAGE ══ */
.landing-hero{{
    text-align:center;padding:80px 40px 60px;
    position:relative;overflow:hidden;
}}
.landing-badge{{
    display:inline-flex;align-items:center;gap:8px;
    background:linear-gradient(135deg,rgba(147,51,234,.18),rgba(0,212,255,.08));
    border:1px solid rgba(147,51,234,.4);border-radius:50px;
    padding:8px 22px;font-size:11px;font-weight:700;letter-spacing:3px;
    color:#c084fc;text-transform:uppercase;margin-bottom:28px;
    animation:glow-pulse 3s ease-in-out infinite;
}}
.landing-title{{
    font-family:'Orbitron','Inter',sans-serif;font-size:clamp(48px,7vw,88px);font-weight:900;
    line-height:1.05;margin-bottom:20px;letter-spacing:-1px;
    background:linear-gradient(135deg,#ffffff 0%,#c084fc 40%,#00d4ff 80%,#ec4899 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    animation:fade-up .7s ease both;
}}
.landing-tagline{{
    font-size:clamp(16px,2.2vw,22px);font-weight:500;color:#64748b;
    margin-bottom:16px;line-height:1.6;max-width:600px;margin-left:auto;margin-right:auto;
    animation:fade-up .8s ease .1s both;
}}
.landing-desc{{
    font-size:15px;color:#475569;line-height:1.8;
    max-width:520px;margin:0 auto 40px;animation:fade-up .9s ease .2s both;
}}
.landing-cta-row{{
    display:flex;gap:16px;justify-content:center;flex-wrap:wrap;
    margin-bottom:60px;animation:fade-up 1s ease .3s both;
}}
.landing-cta-primary{{
    display:inline-flex;align-items:center;gap:10px;
    background:linear-gradient(135deg,#7c3aed,#0891b2);
    color:#fff;font-size:16px;font-weight:700;
    padding:16px 36px;border-radius:50px;border:none;cursor:pointer;
    box-shadow:0 0 32px rgba(147,51,234,.5),0 8px 24px rgba(0,0,0,.4);
    transition:all .25s;text-decoration:none;letter-spacing:.5px;
}}
.landing-cta-primary:hover{{
    background:linear-gradient(135deg,#9333ea,#06b6d4);
    box-shadow:0 0 55px rgba(147,51,234,.7),0 12px 32px rgba(0,0,0,.5);
    transform:translateY(-3px);
}}
.landing-cta-secondary{{
    display:inline-flex;align-items:center;gap:8px;
    background:rgba(255,255,255,.04);border:1px solid rgba(147,51,234,.3);
    color:#94a3b8;font-size:16px;font-weight:600;
    padding:16px 36px;border-radius:50px;cursor:pointer;
    transition:all .25s;text-decoration:none;
}}
.landing-cta-secondary:hover{{
    background:rgba(147,51,234,.1);border-color:rgba(147,51,234,.6);color:#e2e8f0;
    transform:translateY(-3px);
}}
.landing-stats-bar{{
    display:flex;gap:0;justify-content:center;flex-wrap:wrap;
    background:rgba(255,255,255,.02);border:1px solid rgba(147,51,234,.14);
    border-radius:20px;padding:0;margin:0 auto 64px;max-width:760px;overflow:hidden;
}}
.ls-stat{{
    flex:1;min-width:160px;padding:24px 20px;text-align:center;
    border-right:1px solid rgba(147,51,234,.1);position:relative;
}}
.ls-stat:last-child{{border-right:none;}}
.ls-stat::before{{
    content:'';position:absolute;top:0;left:50%;transform:translateX(-50%);
    width:60%;height:2px;background:linear-gradient(90deg,transparent,#9333ea,transparent);
}}
.ls-num{{font-size:32px;font-weight:900;color:#fff;line-height:1;}}
.ls-lbl{{font-size:11px;color:#475569;margin-top:6px;letter-spacing:.5px;}}
.landing-section-label{{
    font-size:10px;font-weight:700;letter-spacing:4px;text-transform:uppercase;
    color:#9333ea;text-align:center;margin-bottom:16px;
}}
.landing-section-title{{
    font-size:clamp(26px,3.5vw,38px);font-weight:800;text-align:center;
    color:#f1f5f9;margin-bottom:10px;line-height:1.2;
}}
.landing-section-sub{{
    font-size:15px;color:#475569;text-align:center;
    max-width:500px;margin:0 auto 48px;line-height:1.7;
}}
.l-feat-card{{
    background:rgba(255,255,255,.025);border:1px solid rgba(147,51,234,.12);
    border-radius:22px;padding:30px 26px;height:100%;
    position:relative;overflow:hidden;transition:all .3s;
}}
.l-feat-card::before{{
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
    opacity:0;transition:opacity .3s;
}}
.l-feat-card:hover{{
    border-color:rgba(147,51,234,.38);
    box-shadow:0 20px 60px rgba(0,0,0,.5),0 0 30px rgba(147,51,234,.08);
    transform:translateY(-5px);
}}
.l-feat-card:hover::before{{opacity:1;}}
.l-feat-icon{{
    width:54px;height:54px;border-radius:16px;
    display:flex;align-items:center;justify-content:center;font-size:24px;
    margin-bottom:18px;
}}
.l-feat-title{{font-size:18px;font-weight:700;color:#f1f5f9;margin-bottom:10px;}}
.l-feat-desc{{font-size:14px;color:#64748b;line-height:1.7;}}
.how-it-works-wrap{{
    display:flex;align-items:flex-start;gap:0;
    max-width:820px;margin:0 auto;
}}
.hiw-step{{
    flex:1;text-align:center;padding:0 16px;
}}
.hiw-num{{
    width:52px;height:52px;border-radius:50%;
    display:flex;align-items:center;justify-content:center;
    font-size:18px;font-weight:900;margin:0 auto 16px;
}}
.hiw-num-1{{background:linear-gradient(135deg,rgba(99,102,241,.3),rgba(99,102,241,.1));border:2px solid rgba(99,102,241,.5);color:#818cf8;box-shadow:0 0 18px rgba(99,102,241,.3);}}
.hiw-num-2{{background:linear-gradient(135deg,rgba(147,51,234,.3),rgba(147,51,234,.1));border:2px solid rgba(147,51,234,.5);color:#c084fc;box-shadow:0 0 18px rgba(147,51,234,.3);}}
.hiw-num-3{{background:linear-gradient(135deg,rgba(0,212,255,.25),rgba(0,212,255,.08));border:2px solid rgba(0,212,255,.45);color:#00d4ff;box-shadow:0 0 18px rgba(0,212,255,.25);}}
.hiw-title{{font-size:16px;font-weight:700;color:#e2e8f0;margin-bottom:8px;}}
.hiw-desc{{font-size:13px;color:#475569;line-height:1.65;}}
.hiw-arrow{{
    flex:none;width:48px;display:flex;align-items:flex-start;
    justify-content:center;padding-top:26px;font-size:22px;
    color:rgba(147,51,234,.35);
}}
.landing-preview-wrap{{
    max-width:680px;margin:0 auto;
    background:linear-gradient(135deg,rgba(10,2,25,.97),rgba(6,1,15,.95));
    border-radius:24px;padding:28px 32px;
    border:1px solid rgba(147,51,234,.3);
    box-shadow:0 0 60px rgba(147,51,234,.18),0 40px 80px rgba(0,0,0,.6);
    position:relative;overflow:hidden;
}}
.landing-preview-wrap::before{{
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent,#9333ea,#00d4ff,#ec4899,transparent);
    animation:shimmer 3s linear infinite;background-size:200% 100%;
}}
.lpv-head{{font-size:12px;letter-spacing:2px;color:#334155;text-transform:uppercase;font-weight:700;margin-bottom:16px;}}
.lpv-row{{display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;flex-wrap:wrap;gap:12px;}}
.lpv-name{{font-size:17px;font-weight:700;color:#e2e8f0;}}
.lpv-badge{{background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.35);border-radius:50px;padding:4px 16px;font-size:12px;font-weight:700;color:#22c55e;}}
.lpv-metrics{{display:flex;gap:24px;flex-wrap:wrap;}}
.lpv-m-val{{font-size:22px;font-weight:800;line-height:1;}}
.lpv-m-lbl{{font-size:10px;color:#475569;margin-top:4px;}}
.landing-footer{{
    text-align:center;padding:48px 40px 32px;
    border-top:1px solid rgba(147,51,234,.1);margin-top:40px;
}}
.lf-links{{display:flex;gap:32px;justify-content:center;margin-bottom:24px;flex-wrap:wrap;}}
.lf-link{{font-size:13px;color:#475569;cursor:default;transition:color .2s;}}
.lf-link:hover{{color:#c084fc;}}
.lf-copy{{font-size:12px;color:#1e293b;}}

/* ══ AUTH PAGE (CENTERED) ══ */
.auth-center-wrap{{
    display:flex;align-items:center;justify-content:center;
    min-height:100vh;padding:20px;
}}
.auth-center-card{{
    background:linear-gradient(160deg,rgba(20,5,40,.96),rgba(8,1,20,.96));
    backdrop-filter:blur(40px);
    border-radius:28px;padding:40px 44px;
    border:1px solid rgba(147,51,234,.35);
    box-shadow:
        0 0 0 1px rgba(0,212,255,.06),
        0 0 80px rgba(147,51,234,.22),
        0 40px 100px rgba(0,0,0,.8),
        inset 0 1px 0 rgba(255,255,255,.06);
    position:relative;overflow:hidden;
    width:100%;max-width:460px;
    animation:fade-up .5s ease both;
}}
.auth-center-card::before{{
    content:'';position:absolute;top:0;left:0;right:0;height:2px;
    background:linear-gradient(90deg,transparent 0%,#9333ea 30%,#00d4ff 60%,#ec4899 80%,transparent 100%);
    animation:shimmer 3s linear infinite;background-size:200% 100%;
}}
.acc-logo-row{{
    display:flex;flex-direction:column;align-items:center;margin-bottom:28px;
}}
.acc-icon{{
    width:64px;height:64px;border-radius:20px;
    background:linear-gradient(135deg,#9333ea,#00d4ff);
    display:flex;align-items:center;justify-content:center;font-size:28px;
    box-shadow:0 0 32px rgba(147,51,234,.55),0 0 64px rgba(0,212,255,.2);
    animation:glow-pulse 3s ease-in-out infinite;margin-bottom:16px;
}}
.acc-name{{
    font-family:'Orbitron',sans-serif;font-size:26px;font-weight:900;
    background:linear-gradient(135deg,#fff,#c084fc,#00d4ff);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    letter-spacing:2px;
}}
.acc-sub{{font-size:12px;color:#334155;letter-spacing:2px;text-transform:uppercase;margin-top:4px;}}
.acc-divider{{height:1px;background:linear-gradient(90deg,transparent,rgba(147,51,234,.3),rgba(0,212,255,.2),transparent);margin:0 0 24px;}}
.acc-form-title{{font-size:22px;font-weight:800;color:#f1f5f9;margin-bottom:4px;}}
.acc-form-sub{{font-size:13px;color:#475569;margin-bottom:20px;}}
.acc-back{{
    display:inline-flex;align-items:center;gap:6px;
    font-size:12px;color:#334155;cursor:pointer;
    transition:color .2s;margin-bottom:20px;
    background:none;border:none;padding:0;
}}
.acc-back:hover{{color:#c084fc;}}
.acc-secure{{
    text-align:center;margin-top:20px;padding-top:16px;
    border-top:1px solid rgba(255,255,255,.04);
    font-size:11px;color:#1e293b;
}}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _password_strength(pwd: str):
    score = 0
    if len(pwd) >= 8: score += 1
    if re.search(r"[A-Z]", pwd): score += 1
    if re.search(r"[0-9]", pwd): score += 1
    if re.search(r"[^A-Za-z0-9]", pwd): score += 1
    labels = ["Very Weak", "Weak", "Fair", "Strong", "Very Strong"]
    colors = ["#ef4444", "#f97316", "#eab308", "#22c55e", "#a855f7"]
    return score, labels[score], colors[score]


def _dark_fig(w=9, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#06010f")
    ax.set_facecolor("none")
    return fig, ax


def _style_ax(ax):
    ax.tick_params(colors="#64748b", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor((1.0, 1.0, 1.0, 0.06))
    ax.grid(alpha=0.07, color="white")


# ══════════════════════════════════════════════════════════════════════════════
# LANDING PAGE  —  full-screen marketing hero
# ══════════════════════════════════════════════════════════════════════════════
def show_landing():
    st.markdown("""
<style>
[data-testid="stSidebar"]{display:none!important;}
[data-testid="stSidebarCollapsedControl"]{display:none!important;}
.block-container{padding-top:0!important;padding-left:0!important;padding-right:0!important;max-width:100%!important;}
</style>""", unsafe_allow_html=True)

    # ── HERO ──────────────────────────────────────────────────────────────────
    st.html(f"""
<div class="landing-hero">
    <div class="landing-badge">✦ &nbsp; AI-POWERED CREDIT INTELLIGENCE &nbsp; ✦</div>
    <div class="landing-title">{APP_SHORT}<br><span style="font-size:.62em;letter-spacing:4px">AI</span></div>
    <div class="landing-tagline">{APP_TAGLINE}</div>
    <div class="landing-desc">
        Instantly evaluate loan applications using a trained Random Forest AI model and a
        financial rule engine — delivering explainable <b style="color:#ef4444">HIGH</b> /
        <b style="color:#22c55e">LOW</b> risk verdicts in milliseconds.
        Built for banks, NBFCs, and credit professionals.
    </div>
</div>
""")

    # CTA Buttons
    _, c1, gap, c2, _ = st.columns([2, 1.5, 0.15, 1.5, 2])
    with c1:
        if st.button("🚀  Get Started  →", key="hero_cta", use_container_width=True):
            st.session_state.page = "auth"
            st.rerun()
    with c2:
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

    # ── STATS BAR ─────────────────────────────────────────────────────────────
    from db import get_application_stats
    stats = get_application_stats()
    total_apps = stats["total_applications"]
    high_risk  = stats["high_risk_count"]
    total_usr  = stats["total_users"]
    high_pct   = (high_risk / total_apps * 100) if total_apps else 0.0

    st.html(f"""
<div style="max-width:820px;margin:0 auto 64px;padding:0 24px">
<div class="landing-stats-bar">
    <div class="ls-stat">
        <div class="ls-num">{total_usr:,}</div>
        <div class="ls-lbl">Registered Users</div>
    </div>
    <div class="ls-stat">
        <div class="ls-num">{total_apps:,}</div>
        <div class="ls-lbl">Applications Processed</div>
    </div>
    <div class="ls-stat">
        <div class="ls-num">{high_pct:.1f}%</div>
        <div class="ls-lbl">High Risk Rate</div>
    </div>
    <div class="ls-stat">
        <div class="ls-num">100</div>
        <div class="ls-lbl">ML Trees in Model</div>
    </div>
</div>
</div>
""")

    # ── FEATURES ──────────────────────────────────────────────────────────────
    st.html("""
<div style="text-align:center;margin-bottom:40px;padding:0 24px">
    <div class="landing-section-label">What We Offer</div>
    <div class="landing-section-title">Everything you need to<br>assess credit risk</div>
    <div class="landing-section-sub">A complete platform combining machine learning, financial analytics, and secure data management.</div>
</div>
""")

    features = [
        ("🤖", "linear-gradient(135deg,rgba(0,212,255,.22),rgba(0,212,255,.06))",
         "linear-gradient(90deg,#00d4ff,#6366f1)",
         "AI Prediction Engine",
         "Random Forest Classifier with 100 decision trees trained on financial patterns. Predicts loan risk with cross-validated accuracy using 5 key features."),
        ("📊", "linear-gradient(135deg,rgba(147,51,234,.22),rgba(147,51,234,.06))",
         "linear-gradient(90deg,#9333ea,#ec4899)",
         "Financial Analytics",
         "Real-time DTI ratio, credit score bands, loan-to-income ratios, score trends and regression analytics — all visualized in interactive dashboards."),
        ("🧮", "linear-gradient(135deg,rgba(20,184,166,.18),rgba(20,184,166,.05))",
         "linear-gradient(90deg,#14b8a6,#22c55e)",
         "EMI Calculator",
         "Full amortization schedules, principal vs interest breakdowns, balance projections and interactive charts. Plan repayments with precision."),
        ("🤝", "linear-gradient(135deg,rgba(99,102,241,.2),rgba(99,102,241,.06))",
         "linear-gradient(90deg,#6366f1,#a855f7)",
         "AI Chat Assistant",
         "Conversational risk explainer with context-aware answers. Understand exactly why a decision was made and what can be improved."),
        ("📈", "linear-gradient(135deg,rgba(245,158,11,.18),rgba(245,158,11,.05))",
         "linear-gradient(90deg,#f59e0b,#ef4444)",
         "Trend Analytics",
         "Risk distribution charts, income vs loan regression, score histograms, DTI trends, and downloadable CSV exports for audit trails."),
        ("🔒", "linear-gradient(135deg,rgba(34,197,94,.16),rgba(34,197,94,.04))",
         "linear-gradient(90deg,#22c55e,#06b6d4)",
         "Bank-grade Security",
         "PBKDF2-SHA256 password hashing with 100,000 iterations, parameterized SQL queries preventing injection, and role-based access control."),
    ]

    r1 = st.columns(3, gap="large")
    r2 = st.columns(3, gap="large")
    st.markdown('<div style="padding:0 24px">', unsafe_allow_html=True)
    for col, (icon, icon_bg, bar_grad, title, desc) in zip(r1 + r2, features):
        with col:
            st.html(f"""
<div class="l-feat-card" style="margin-bottom:20px">
    <div style="position:absolute;top:0;left:0;right:0;height:2px;background:{bar_grad};opacity:0;transition:opacity .3s" class="card-top-bar"></div>
    <div class="l-feat-icon" style="background:{icon_bg}">{icon}</div>
    <div class="l-feat-title">{title}</div>
    <div class="l-feat-desc">{desc}</div>
</div>
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── HOW IT WORKS ──────────────────────────────────────────────────────────
    st.html("""
<div style="text-align:center;margin-bottom:40px;padding:0 24px">
    <div class="landing-section-label">The Process</div>
    <div class="landing-section-title">From application to verdict<br>in seconds</div>
    <div class="landing-section-sub">Three simple steps powered by AI, financial rules, and real-time computation.</div>
</div>
<div class="how-it-works-wrap" style="padding:0 24px">
    <div class="hiw-step">
        <div class="hiw-num hiw-num-1">1</div>
        <div class="hiw-title">Enter Details</div>
        <div class="hiw-desc">Provide applicant age, annual income, loan amount, credit score, and monthly EMI through our structured form.</div>
    </div>
    <div class="hiw-arrow">→</div>
    <div class="hiw-step">
        <div class="hiw-num hiw-num-2">2</div>
        <div class="hiw-title">AI Analyzes</div>
        <div class="hiw-desc">Random Forest ML model runs in parallel with a rule engine — evaluating DTI ratios, credit bands, and loan-to-income multiples.</div>
    </div>
    <div class="hiw-arrow">→</div>
    <div class="hiw-step">
        <div class="hiw-num hiw-num-3">3</div>
        <div class="hiw-title">Get Verdict</div>
        <div class="hiw-desc">Receive a clear HIGH / LOW risk decision with full reasoning, financial score out of 100, and improvement recommendations.</div>
    </div>
</div>
""")

    st.markdown("<div style='height:64px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── SAMPLE OUTPUT ─────────────────────────────────────────────────────────
    st.html("""
<div style="text-align:center;margin-bottom:40px;padding:0 24px">
    <div class="landing-section-label">Live Preview</div>
    <div class="landing-section-title">See what a real assessment looks like</div>
</div>
""")

    _, prev_col, _ = st.columns([1.5, 5, 1.5])
    with prev_col:
        st.html("""
<div class="landing-preview-wrap">
    <div class="lpv-head">⟡ &nbsp; SAMPLE ASSESSMENT OUTPUT &nbsp; ⟡</div>
    <div class="lpv-row">
        <div class="lpv-name">Ravi Sharma &nbsp;·&nbsp; Home Loan &nbsp;·&nbsp; ₹35,00,000</div>
        <div class="lpv-badge">✓ LOW RISK</div>
    </div>
    <div class="lpv-metrics">
        <div>
            <div class="lpv-m-val" style="color:#a855f7">82<span style="font-size:14px;color:#475569">/100</span></div>
            <div class="lpv-m-lbl">Financial Score</div>
        </div>
        <div>
            <div class="lpv-m-val" style="color:#00d4ff">740</div>
            <div class="lpv-m-lbl">Credit Score</div>
        </div>
        <div>
            <div class="lpv-m-val" style="color:#2dd4bf">22.4%</div>
            <div class="lpv-m-lbl">DTI Ratio</div>
        </div>
        <div>
            <div class="lpv-m-val" style="color:#fbbf24">2.1×</div>
            <div class="lpv-m-lbl">Loan / Income</div>
        </div>
        <div>
            <div class="lpv-m-val" style="color:#818cf8">₹28,500</div>
            <div class="lpv-m-lbl">Monthly EMI</div>
        </div>
    </div>
    <div style="margin-top:20px;padding-top:16px;border-top:1px solid rgba(147,51,234,.12)">
        <div style="font-size:11px;color:#334155;letter-spacing:1px;text-transform:uppercase;margin-bottom:10px">Analysis Reasoning</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap">
            <span style="background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.25);border-radius:50px;padding:3px 12px;font-size:11px;color:#22c55e">✓ Credit score 700+ (Very Good)</span>
            <span style="background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.25);border-radius:50px;padding:3px 12px;font-size:11px;color:#22c55e">✓ DTI under 30% (Excellent)</span>
            <span style="background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.25);border-radius:50px;padding:3px 12px;font-size:11px;color:#22c55e">✓ Loan within 3× income</span>
            <span style="background:rgba(0,212,255,.08);border:1px solid rgba(0,212,255,.22);border-radius:50px;padding:3px 12px;font-size:11px;color:#38bdf8">ML Model: LOW (conf. 94%)</span>
        </div>
    </div>
</div>
""")

    st.markdown("<div style='height:64px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── FOOTER CTA ────────────────────────────────────────────────────────────
    st.html("""
<div class="landing-footer">
    <div style="font-size:28px;font-weight:800;color:#e2e8f0;margin-bottom:8px">Ready to get started?</div>
    <div style="font-size:14px;color:#475569;margin-bottom:28px">Join banks and NBFCs already using RiskOra AI for smarter lending decisions.</div>
</div>
""")

    _, fc_col, _ = st.columns([3, 2, 3])
    with fc_col:
        if st.button("🚀  Create Free Account", key="footer_cta", use_container_width=True):
            st.session_state.page = "auth"
            st.session_state.auth_tab = "signup"
            st.rerun()

    st.html("""
<div style="text-align:center;padding:24px 40px 32px">
    <div style="font-size:12px;color:#1e293b">
        © 2026 RiskOra AI · Built with Streamlit &amp; scikit-learn · All rights reserved
    </div>
</div>
""")


# ══════════════════════════════════════════════════════════════════════════════
# AUTH PAGE  —  centered glassmorphism card
# ══════════════════════════════════════════════════════════════════════════════
def show_auth():
    st.markdown("""
<style>
[data-testid="stSidebar"]{display:none!important;}
[data-testid="stSidebarCollapsedControl"]{display:none!important;}
.block-container{padding-top:2rem!important;max-width:600px!important;margin:0 auto!important;}
</style>""", unsafe_allow_html=True)

    # Back to home
    if st.button("← Back to Home", key="auth_back"):
        st.session_state.page = "landing"
        st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Card header
    st.html(f"""
<div class="auth-center-card">
    <div class="acc-logo-row">
        <div class="acc-icon">🔮</div>
        <div class="acc-name">{APP_SHORT}</div>
        <div class="acc-sub">AI Credit Intelligence</div>
    </div>
    <div class="acc-divider"></div>
</div>
""")

    # Tab toggle
    ta, tb = st.columns(2, gap="small")
    is_login  = st.session_state.auth_tab == "login"
    is_signup = st.session_state.auth_tab == "signup"

    with ta:
        st.markdown(f'<div class="{"auth-tab-active" if is_login else "auth-tab-inactive"}">', unsafe_allow_html=True)
        if st.button("🔑  Login", key="tab_login_btn"):
            st.session_state.auth_tab = "login"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with tb:
        st.markdown(f'<div class="{"auth-tab-active" if is_signup else "auth-tab-inactive"}">', unsafe_allow_html=True)
        if st.button("📝  Sign Up", key="tab_signup_btn"):
            st.session_state.auth_tab = "signup"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── LOGIN ──────────────────────────────────────────────────────────────────
    if is_login:
        st.html("""
<div style="margin-bottom:18px">
    <div class="acc-form-title">Welcome back 👋</div>
    <div class="acc-form-sub">Sign in to your RiskOra dashboard</div>
</div>
""")
        with st.form("login_form"):
            username  = st.text_input("Username", placeholder="Enter your username")
            password  = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In  →")

        if submitted:
            if not username.strip() or not password.strip():
                st.error("Please enter both username and password.")
            else:
                with st.spinner("Authenticating…"):
                    role = validate_login(username.strip(), password)
                if role:
                    st.session_state.logged_in = True
                    st.session_state.role      = role
                    st.session_state.username  = username.strip()
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")

        st.html("""
<div class="acc-secure">🔒 PBKDF2-SHA256 · 100,000 iterations · Never stored in plain text</div>
""")

    # ── SIGN UP ────────────────────────────────────────────────────────────────
    else:
        st.html("""
<div style="margin-bottom:18px">
    <div class="acc-form-title">Create account 🚀</div>
    <div class="acc-form-sub">Join RiskOra and start analyzing credit risk</div>
</div>
""")
        new_user = st.text_input("Username", placeholder="Choose a username", key="su_user")
        new_pass = st.text_input("Password", type="password", placeholder="Min 6 characters", key="su_pass")

        if new_pass:
            strength, pwd_label, pwd_color = _password_strength(new_pass)
            dim  = "rgba(255,255,255,.08)"
            segs = "".join(
                f'<div class="pwd-seg" style="background:{pwd_color if i <= strength - 1 else dim}"></div>'
                for i in range(4)
            )
            st.html(f"""
<div style="margin-top:-6px;margin-bottom:12px">
    <div class="pwd-bar-wrap">{segs}</div>
    <div style="font-size:11px;color:{pwd_color};margin-top:5px;font-weight:600">Strength: {pwd_label}</div>
</div>
""")

        su_role = st.selectbox("Account Role", ["user", "admin"], key="su_role")

        if st.button("Create Account  →", key="su_btn"):
            if not new_user.strip() or not new_pass.strip():
                st.error("Username and password are required.")
            elif len(new_pass) < 6:
                st.error("Password must be at least 6 characters.")
            elif add_user(new_user.strip(), new_pass, su_role):
                st.success("✅ Account created! Click Login to sign in.")
            else:
                st.error("That username is already taken.")

        st.html("""
<div class="acc-secure">Your password is hashed and never stored in plain text.</div>
""")


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
def show_home():
    stats      = get_application_stats()
    total_apps = stats["total_applications"]
    high_risk  = stats["high_risk_count"]
    total_usr  = stats["total_users"]
    high_pct   = (high_risk / total_apps * 100) if total_apps else 0.0

    left, right = st.columns([6, 4], gap="large")
    with left:
        st.html(f"""
        <div style="padding-top:10px">
            <div class="ai-chip">🔮 AI POWERED</div>
            <div class="htitle">{APP_SHORT}<br>Credit Intelligence</div>
            <p style="font-size:16px;color:#64748b;line-height:1.75;margin-bottom:28px;max-width:500px">
                Instantly evaluate loan applications using a trained Random Forest
                AI model and a financial rule engine — delivering explainable HIGH / LOW
                risk verdicts backed by credit score, DTI, and loan-to-income analysis.
            </p>
        </div>
        <div class="stat-row">
            <div class="stat-c">
                <div style="font-size:20px;margin-bottom:8px">👥</div>
                <div class="stat-val">{total_usr:,}</div>
                <div class="stat-lbl">Registered Users</div>
                <div class="stat-up">▲ Active</div>
            </div>
            <div class="stat-c">
                <div style="font-size:20px;margin-bottom:8px">📋</div>
                <div class="stat-val">{total_apps:,}</div>
                <div class="stat-lbl">Applications</div>
                <div class="stat-up">▲ Growing</div>
            </div>
            <div class="stat-c">
                <div style="font-size:20px;margin-bottom:8px">⚠️</div>
                <div class="stat-val">{high_pct:.1f}%</div>
                <div class="stat-lbl">High Risk Rate</div>
                <div class="{'stat-dn' if high_pct>40 else 'stat-up'}">
                    {'▲ Elevated' if high_pct>40 else '▼ Controlled'}
                </div>
            </div>
        </div>
        """)

    with right:
        st.markdown('<div style="display:flex;justify-content:center;margin-bottom:16px"><div class="ai-orb">🔮</div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.html("""
            <div class="float-card">
                <div class="fc-lbl">Credit Score</div>
                <div class="fc-val">720</div>
                <div class="fc-sub">✦ Excellent</div>
            </div>
            <div class="float-card">
                <div class="fc-lbl">DTI Ratio</div>
                <div class="fc-val">24.6%</div>
                <div class="fc-sub">✦ Healthy</div>
            </div>
            """)
        with c2:
            st.html("""
            <div class="float-card">
                <div class="fc-lbl">Risk Decision</div>
                <div class="fc-val" style="color:#22c55e;font-size:15px">LOW RISK ✓</div>
                <div class="fc-sub" style="color:#22c55e">Approved</div>
            </div>
            <div class="float-card">
                <div class="fc-lbl">AI Score</div>
                <div class="fc-val">87/100</div>
                <div class="fc-sub">✦ Strong</div>
            </div>
            """)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns(3, gap="large")
    features = [
        ("🤖","linear-gradient(135deg,rgba(0,212,255,.2),rgba(0,212,255,.05))","AI Prediction Engine",
         "Random Forest Classifier trained on financial patterns evaluates 5 features and predicts loan risk with cross-validated accuracy."),
        ("📊","linear-gradient(135deg,rgba(147,51,234,.2),rgba(147,51,234,.05))","Financial Analytics",
         "DTI ratio, credit score bands, loan-to-income ratios, score trends and regression analytics in real time."),
        ("🔒","linear-gradient(135deg,rgba(34,197,94,.2),rgba(34,197,94,.05))","Secure by Design",
         "PBKDF2-SHA256 password hashing, parameterized SQL queries, and role-based access control throughout."),
    ]
    for col, (icon, icon_bg, title, desc) in zip([fc1, fc2, fc3], features):
        with col:
            st.html(f"""
            <div class="feat-card">
                <div class="feat-icon" style="background:{icon_bg}">{icon}</div>
                <div class="feat-title">{title}</div>
                <div class="feat-desc">{desc}</div>
            </div>
            """)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.html(f"""
    <div class="session-card">
        <div class="slabel" style="margin-bottom:16px">Current Session</div>
        <div style="display:flex;align-items:center;gap:50px;flex-wrap:wrap">
            <div><div style="font-size:10px;color:#475569;letter-spacing:1px;text-transform:uppercase">👤 User</div>
                 <div style="font-size:16px;font-weight:700;color:#e2e8f0;margin-top:4px">{st.session_state.username}</div></div>
            <div><div style="font-size:10px;color:#475569;letter-spacing:1px;text-transform:uppercase">🛡️ Role</div>
                 <div style="margin-top:4px"><span class="{'sb-role-admin' if st.session_state.role=='admin' else 'sb-role-user'}">{st.session_state.role}</span></div></div>
            <div><div style="font-size:10px;color:#475569;letter-spacing:1px;text-transform:uppercase">💾 Storage</div>
                 <div style="font-size:16px;font-weight:700;color:#e2e8f0;margin-top:4px">SQLite Database</div></div>
            <div><div style="font-size:10px;color:#475569;letter-spacing:1px;text-transform:uppercase">🤖 Engine</div>
                 <div style="font-size:16px;font-weight:700;color:#e2e8f0;margin-top:4px">Random Forest + Rules</div></div>
        </div>
    </div>
    """)

    try:
        importances = get_feature_importances()
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="slabel" style="margin-bottom:10px">Model Feature Importance</div>', unsafe_allow_html=True)
        fig, ax = _dark_fig(9, 2.8)
        colors = ["#00d4ff","#9333ea","#f97316","#22c55e","#f43f5e"]
        ax.barh(list(importances.keys()), list(importances.values()), color=colors, height=0.45)
        ax.set_xlabel("Importance", color="#475569", fontsize=11)
        _style_ax(ax); fig.tight_layout()
        st.pyplot(fig); plt.close(fig)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# LOAN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def show_apply_loan():
    st.markdown('<div class="page-title">Loan Risk Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Complete the form — AI will assess risk instantly with full explanation</div>', unsafe_allow_html=True)

    st.markdown('<div class="gc">', unsafe_allow_html=True)
    with st.form("loan_form"):
        applicant_name = st.text_input("Applicant Full Name", value=st.session_state.username)
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown('<div class="slabel">Personal &amp; Financial</div>', unsafe_allow_html=True)
            applicant_email = st.text_input("Email (optional)", placeholder="email@example.com")
            age             = st.number_input("Age", min_value=18, max_value=65, value=30, help="18–65 years")
            income          = st.number_input("Annual Income (₹)", min_value=0.0, value=500000.0, step=10000.0)
            credit_score    = st.number_input("Credit Score (300–900)", min_value=300, max_value=900, value=700)
        with col2:
            st.markdown('<div class="slabel">Loan Details</div>', unsafe_allow_html=True)
            phone   = st.text_input("Phone (optional)", placeholder="+91 XXXXX XXXXX")
            loan    = st.number_input("Loan Amount (₹)", min_value=0.0, value=100000.0, step=10000.0)
            emi     = st.number_input("Monthly EMI (₹)", min_value=0.0, value=10000.0, step=500.0)
            purpose = st.selectbox("Loan Purpose", ["Personal","Home","Education","Vehicle","Business","Medical","Other"])
        notes     = st.text_area("Additional Notes (optional)", placeholder="Any relevant information…")
        submitted = st.form_submit_button("⚡  Analyze Risk Now")
    st.markdown('</div>', unsafe_allow_html=True)

    if not submitted:
        return

    if not applicant_name.strip():
        st.error("Applicant name is required."); return
    if applicant_email and not validate_email(applicant_email):
        st.error("Please enter a valid email address."); return
    if phone and not validate_phone(phone):
        st.error("Please enter a valid phone number."); return
    msg = validate(age, income, loan, credit_score, emi)
    if msg != "OK":
        st.error(f"Validation error: {msg}"); return

    with st.spinner("🔮 AI analyzing application…"):
        dti        = calculate_dti(income, emi)
        score      = financial_score(credit_score, dti)
        ml_result  = predict([age, income, loan, credit_score, emi])
        rule_result, reasons = rule_override(age, credit_score, dti, loan, income)
    final_risk = "HIGH RISK" if (rule_result == "HIGH" or ml_result == 1) else "LOW RISK"

    insert_application({
        "applicant_name": applicant_name.strip(), "applicant_email": applicant_email.strip(),
        "phone": phone.strip(), "purpose": purpose, "notes": notes.strip(),
        "age": age, "income": income, "loan": loan, "credit_score": credit_score,
        "emi": emi, "dti": dti, "score": score, "risk": final_risk,
        "submitted_by": st.session_state.username,
    })
    st.session_state["data"] = {
        "credit_score": credit_score, "dti": dti, "score": score, "risk": final_risk,
        "applicant_name": applicant_name.strip(), "purpose": purpose, "reasons": reasons,
        "income": income, "loan": loan, "emi": emi,
    }
    st.session_state.chat_history = []
    icon = "🔴" if final_risk == "HIGH RISK" else "🟢"
    st.toast(f"{icon} Analysis complete — {final_risk}", icon="🔮")
    _show_result_card(applicant_name.strip(), final_risk, rule_result,
                      score, credit_score, dti, reasons, ml_result, loan, income)


def _show_result_card(name, final_risk, rule_result, score, cs, dti, reasons, ml_result, loan, income):
    rb = "rb-high" if final_risk == "HIGH RISK" else "rb-low"
    lr = loan / income if income else 0.0
    st.html(f"""
    <div class="glow-card" style="margin-top:24px">
        <div class="slabel">Analysis Result</div>
        <div style="font-size:14px;color:#64748b;margin-bottom:8px">
            Applicant: <b style="color:#e2e8f0">{name}</b>
        </div>
        <div style="margin:8px 0 18px"><span class="{rb}">{final_risk}</span></div>
        <div class="mrow">
            <div class="mbox">
                <div class="mbox-lbl">Financial Score</div>
                <div class="mbox-val">{score}<span style="font-size:13px;color:#475569">/100</span></div>
                <div class="mbox-sub">{score_label(score)}</div>
            </div>
            <div class="mbox">
                <div class="mbox-lbl">Credit Score</div>
                <div class="mbox-val">{cs}</div>
                <div class="mbox-sub">{credit_score_band(cs)}</div>
            </div>
            <div class="mbox">
                <div class="mbox-lbl">DTI Ratio</div>
                <div class="mbox-val">{fmt_percent(dti)}</div>
                <div class="mbox-sub">{dti_band(dti)}</div>
            </div>
            <div class="mbox">
                <div class="mbox-lbl">Loan / Income</div>
                <div class="mbox-val">{lr:.1f}<span style="font-size:13px;color:#475569">×</span></div>
                <div class="mbox-sub">{"Safe" if lr <= 4 else "Elevated"}</div>
            </div>
        </div>
    </div>
    """)

    c1, c2 = st.columns(2, gap="large")
    ml_col  = "#ef4444" if ml_result == 1 else "#22c55e"
    ml_lbl  = "HIGH RISK" if ml_result == 1 else "LOW RISK"
    rl_col  = "#ef4444" if rule_result == "HIGH" else ("#f97316" if rule_result == "CAUTION" else "#22c55e")
    with c1:
        st.html(f"""<div class="gc">
            <div class="slabel">ML Model</div>
            <div style="font-size:12px;color:#475569;margin-bottom:8px">Random Forest Classifier</div>
            <div style="color:{ml_col};font-size:20px;font-weight:800">{ml_lbl}</div>
        </div>""")
    with c2:
        st.html(f"""<div class="gc">
            <div class="slabel">Rule Engine</div>
            <div style="font-size:12px;color:#475569;margin-bottom:8px">Business Rule Override</div>
            <div style="color:{rl_col};font-size:20px;font-weight:800">{rule_result}</div>
        </div>""")

    if reasons:
        for r in reasons:
            st.warning(f"⚠ {r}")
    st.info("✅ Saved. Open **Dashboard** to track history or **AI Assistant** for insights.")


# ══════════════════════════════════════════════════════════════════════════════
# EMI CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
def show_emi_calculator():
    st.markdown('<div class="page-title">EMI Calculator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Monthly instalment, total interest, and full amortization schedule</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<div class="gc">', unsafe_allow_html=True)
        st.markdown('<div class="slabel">Loan Parameters</div>', unsafe_allow_html=True)
        principal   = st.number_input("Loan Amount (₹)", min_value=10000.0, value=500000.0, step=10000.0, key="emi_p")
        annual_rate = st.number_input("Annual Interest Rate (%)", min_value=0.1, max_value=36.0, value=10.5, step=0.1, key="emi_r")
        tenure_yrs  = st.number_input("Loan Tenure (Years)", min_value=1, max_value=30, value=5, step=1, key="emi_t")
        st.button("🧮  Recalculate", key="emi_calc")
        st.markdown('</div>', unsafe_allow_html=True)

    tenure_months = tenure_yrs * 12
    r = (annual_rate / 100) / 12
    if r == 0:
        emi_val = principal / tenure_months
    else:
        emi_val = principal * r * (1 + r) ** tenure_months / ((1 + r) ** tenure_months - 1)
    total_pay = emi_val * tenure_months
    total_int = total_pay - principal

    with right:
        st.html(f"""
        <div class="emi-card">
            <div class="slabel" style="margin-bottom:16px">Monthly EMI</div>
            <div style="font-size:48px;font-weight:900;color:#c084fc;line-height:1">₹{emi_val:,.0f}</div>
            <div style="font-size:13px;color:#475569;margin-top:6px">per month for {tenure_yrs} year{'s' if tenure_yrs>1 else ''}</div>
            <div class="divider" style="margin:16px 0"></div>
            <div style="display:flex;gap:28px;flex-wrap:wrap">
                <div><div class="slabel">Principal</div><div style="font-size:18px;font-weight:700;color:#e2e8f0">₹{principal:,.0f}</div></div>
                <div><div class="slabel">Total Interest</div><div style="font-size:18px;font-weight:700;color:#f97316">₹{total_int:,.0f}</div></div>
                <div><div class="slabel">Total Payment</div><div style="font-size:18px;font-weight:700;color:#22c55e">₹{total_pay:,.0f}</div></div>
            </div>
        </div>
        """)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col_pie, col_line = st.columns(2, gap="large")
    with col_pie:
        st.markdown('<div class="slabel" style="margin-bottom:10px">Principal vs Interest</div>', unsafe_allow_html=True)
        fig, ax = _dark_fig(4.5, 4.5)
        _, _, autotexts = ax.pie(
            [principal, total_int], labels=["Principal","Interest"],
            colors=["#9333ea","#f97316"], explode=[0.04,0.04],
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"linewidth":1.5,"edgecolor":"#06010f"},
            textprops={"color":"#94a3b8","fontsize":12},
        )
        for at in autotexts:
            at.set_color("white"); at.set_fontweight("bold"); at.set_fontsize(12)
        ax.set_title("Breakdown", color="#475569", pad=10, fontsize=12)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with col_line:
        st.markdown('<div class="slabel" style="margin-bottom:10px">Loan Balance Over Time</div>', unsafe_allow_html=True)
        balance = principal; balances = [balance]
        for _ in range(tenure_months):
            ip = balance * r; pp = emi_val - ip
            balance = max(0.0, balance - pp); balances.append(balance)
        xs = list(range(tenure_months + 1))
        fig, ax = _dark_fig(5, 4.5)
        ax.plot(xs, balances, color="#9333ea", linewidth=2.5)
        ax.fill_between(xs, balances, alpha=0.12, color="#9333ea")
        ax.set_xlabel("Month", color="#475569", fontsize=11)
        ax.set_ylabel("Balance (₹)", color="#475569", fontsize=11)
        ax.set_title("Amortization", color="#475569", fontsize=12)
        _style_ax(ax); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.markdown('<div class="slabel" style="margin:24px 0 10px">Yearly Amortization Schedule</div>', unsafe_allow_html=True)
    rows = []; balance = principal
    for yr in range(1, tenure_yrs + 1):
        yr_p = yr_i = 0.0
        for _ in range(12):
            ip = balance * r; pp = min(emi_val - ip, balance)
            yr_i += ip; yr_p += pp; balance = max(0.0, balance - pp)
        rows.append({"Year": yr, "Principal Paid ₹": f"{yr_p:,.0f}",
                     "Interest Paid ₹": f"{yr_i:,.0f}", "Remaining Balance ₹": f"{balance:,.0f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def show_dashboard():
    st.markdown('<div class="page-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Your personal loan application history and analytics</div>', unsafe_allow_html=True)

    history = get_user_applications(st.session_state.username)
    if not history:
        st.markdown('<div class="gc" style="text-align:center;padding:48px"><div style="font-size:40px;margin-bottom:12px">📭</div><div style="color:#475569">No applications yet. Submit a loan analysis first.</div></div>', unsafe_allow_html=True)
        return

    latest     = history[0]
    high_count = sum(1 for h in history if h["risk"] == "HIGH RISK")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Applications", len(history))
    m2.metric("Latest Score",       f"{latest['score']}/100")
    m3.metric("Latest Risk",        latest["risk"])
    m4.metric("High Risk Count",    high_count)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    f1, f2 = st.columns([3, 1], gap="large")
    with f1:
        search = st.text_input("🔍  Search", placeholder="Search by name, purpose, risk…", label_visibility="collapsed")
    with f2:
        risk_f = st.selectbox("Risk", ["All","HIGH RISK","LOW RISK"], label_visibility="collapsed")

    filtered = history
    if search:
        kw = search.lower()
        filtered = [h for h in filtered if kw in h.get("applicant_name","").lower()
                    or kw in h.get("purpose","").lower() or kw in h.get("risk","").lower()]
    if risk_f != "All":
        filtered = [h for h in filtered if h["risk"] == risk_f]

    history_df = pd.DataFrame(history)
    filt_df    = pd.DataFrame(filtered) if filtered else pd.DataFrame()

    col_pie, col_line = st.columns(2, gap="large")
    with col_pie:
        st.markdown('<div class="slabel" style="margin:16px 0 8px">Risk Distribution</div>', unsafe_allow_html=True)
        low_count = len(history) - high_count
        fig, ax = _dark_fig(4.5, 4.5)
        _, _, autotexts = ax.pie(
            [low_count, high_count], labels=["Low Risk","High Risk"],
            colors=["#22c55e","#ef4444"], explode=[0.04,0.04],
            autopct="%1.0f%%", startangle=90,
            wedgeprops={"linewidth":1.5,"edgecolor":"#06010f"},
            textprops={"color":"#94a3b8","fontsize":12},
        )
        for at in autotexts:
            at.set_color("white"); at.set_fontweight("bold"); at.set_fontsize(13)
        ax.set_title("By Risk", color="#475569", pad=10, fontsize=12)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with col_line:
        st.markdown('<div class="slabel" style="margin:16px 0 8px">Score Trend</div>', unsafe_allow_html=True)
        if len(history_df) > 1:
            fig, ax = _dark_fig(5, 4.5)
            scores = list(reversed(history_df["score"].tolist()))
            xs = range(1, len(scores)+1)
            ax.plot(xs, scores, color="#9333ea", linewidth=2.5, marker="o",
                    markersize=7, markerfacecolor="#06010f",
                    markeredgecolor="#9333ea", markeredgewidth=2)
            ax.fill_between(xs, scores, alpha=0.1, color="#9333ea")
            ax.set_xlabel("Application #", color="#475569", fontsize=11)
            ax.set_ylabel("Score",          color="#475569", fontsize=11)
            ax.set_ylim(0, 100); _style_ax(ax)
            ax.set_title("Score over time", color="#475569", fontsize=12)
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)
        else:
            st.info("Submit more applications to see the score trend.")

    st.markdown('<div class="slabel" style="margin:24px 0 8px">Loan Amount vs Financial Score</div>', unsafe_allow_html=True)
    render_regression_chart(history_df, "loan", "score",
        "Loan Amount vs Financial Score", "Loan Amount (₹)", "Financial Score")

    st.markdown(f'<div class="slabel" style="margin:24px 0 8px">Applications ({len(filtered)} of {len(history)})</div>', unsafe_allow_html=True)
    if not filt_df.empty:
        disp = filt_df[["id","applicant_name","purpose","loan","credit_score","dti","score","risk","created_at"]].copy()
        disp.columns = ["ID","Applicant","Purpose","Loan ₹","Credit Score","DTI %","Score","Risk","Submitted At"]
        st.dataframe(disp, use_container_width=True)
        csv = disp.to_csv(index=False).encode("utf-8")
        st.download_button("⬇  Download CSV", data=csv,
            file_name=f"{st.session_state.username}_applications.csv", mime="text/csv")
    else:
        st.info("No applications match your filter.")


def render_regression_chart(df, x_col, y_col, title, x_label, y_label):
    cdf = df[[x_col, y_col]].dropna()
    if len(cdf) < 2:
        st.info("At least two applications needed for regression."); return
    x = cdf[x_col].astype(float).to_numpy()
    y = cdf[y_col].astype(float).to_numpy()
    if x.max() == x.min():
        st.info("All x-values identical — cannot draw regression line."); return
    slope, intercept = np.polyfit(x, y, 1)
    rx = np.linspace(x.min(), x.max(), 200); ry = slope * rx + intercept
    fig, ax = _dark_fig(9, 4)
    ax.scatter(x, y, color="#9333ea", s=70, alpha=0.85, zorder=3,
               label="Applications", edgecolors="#06010f", linewidths=0.5)
    ax.plot(rx, ry, color="#f97316", linewidth=2.5, label="Trend")
    ax.set_title(title, color="#475569", fontsize=12, pad=10)
    ax.set_xlabel(x_label, color="#475569", fontsize=11)
    ax.set_ylabel(y_label, color="#475569", fontsize=11)
    ax.legend(facecolor="#0f172a", edgecolor=(147/255,51/255,234/255,0.3), labelcolor="#94a3b8")
    _style_ax(ax); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.caption(f"Equation: {y_label} = {slope:.4f} × {x_label} + {intercept:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# AI ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
def show_chatbot():
    st.markdown('<div class="page-title">AI Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Intelligent analysis and advice for your latest loan application</div>', unsafe_allow_html=True)

    if "data" not in st.session_state:
        st.markdown('<div class="gc" style="text-align:center;padding:48px"><div style="font-size:40px;margin-bottom:12px">🔮</div><div style="color:#475569">Analyze a loan application first — I need context to assist you.</div></div>', unsafe_allow_html=True)
        return

    data = st.session_state["data"]
    risk = data.get("risk","")
    rb   = "rb-low" if risk == "LOW RISK" else "rb-high"
    st.html(f"""
    <div class="gc">
        <div class="slabel">Active Context</div>
        <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-top:8px">
            <div style="font-size:16px;font-weight:700;color:#e2e8f0">{data.get('applicant_name','')}</div>
            <span class="{rb}" style="font-size:13px;padding:5px 16px">{risk}</span>
            <span style="color:#475569;font-size:13px">Score: <b style="color:#c084fc">{data.get('score',0)}/100</b></span>
            <span style="color:#475569;font-size:13px">DTI: <b style="color:#00d4ff">{fmt_percent(data.get('dti',0))}</b></span>
            <span style="color:#475569;font-size:13px">Credit: <b style="color:#2dd4bf">{data.get('credit_score',0)}</b></span>
        </div>
    </div>
    """)

    history = st.session_state.chat_history
    if history:
        st.markdown('<div class="gc" style="max-height:420px;overflow-y:auto;padding:16px">', unsafe_allow_html=True)
        for msg in history:
            if msg["role"] == "user":
                st.html(f"""
                <div class="chat-user-wrap">
                    <div class="chat-user">{msg["content"]}</div>
                    <div class="chat-avatar-u">👤</div>
                </div>""")
            else:
                st.html(f"""
                <div class="chat-bot-wrap">
                    <div class="chat-avatar-b">🔮</div>
                    <div class="chat-bot">{msg["content"]}</div>
                </div>""")
        st.markdown('</div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        fi, fb = st.columns([6, 1], gap="small")
        with fi:
            question = st.text_input("", placeholder='Ask something… e.g. "Why is this high risk?"',
                                     label_visibility="collapsed", key="chat_q")
        with fb:
            send = st.form_submit_button("Send ➤")

    if send and question.strip():
        resp = chatbot_response(data, question.strip())
        st.session_state.chat_history.append({"role":"user",      "content": question.strip()})
        st.session_state.chat_history.append({"role":"assistant",  "content": resp})
        st.rerun()

    st.markdown('<div class="slabel" style="margin:20px 0 10px">Quick Questions</div>', unsafe_allow_html=True)
    quick = ["Why is this result high or low risk?","How can the applicant improve?",
             "Can the applicant afford this loan?","What does the credit score mean?","What is the DTI ratio?"]
    qa, qb = st.columns(2, gap="medium")
    for i, q in enumerate(quick):
        col = qa if i % 2 == 0 else qb
        with col:
            if st.button(q, key=f"qk_{i}"):
                resp = chatbot_response(data, q)
                st.session_state.chat_history.append({"role":"user",      "content": q})
                st.session_state.chat_history.append({"role":"assistant", "content": resp})
                st.rerun()

    if history:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        if st.button("🗑  Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []; st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN PANEL
# ══════════════════════════════════════════════════════════════════════════════
def show_admin_panel():
    st.markdown('<div class="page-title">Admin Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">System management, user control and full analytics</div>', unsafe_allow_html=True)

    stats = get_application_stats()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registered Users",   stats["total_users"])
    m2.metric("Total Applications", stats["total_applications"])
    m3.metric("High Risk",          stats["high_risk_count"])
    m4.metric("Low Risk",           stats["low_risk_count"])

    st.markdown('<div class="slabel" style="margin:28px 0 12px">Create New User</div>', unsafe_allow_html=True)
    st.markdown('<div class="gc">', unsafe_allow_html=True)
    with st.form("admin_add_user"):
        a1, a2, a3 = st.columns(3, gap="large")
        with a1: au = st.text_input("Username")
        with a2: ap = st.text_input("Password", type="password")
        with a3: ar = st.selectbox("Role", ["user","admin"])
        if st.form_submit_button("Create User"):
            if not au.strip() or not ap.strip():
                st.error("Username and password required.")
            elif len(ap) < 6:
                st.error("Password must be at least 6 characters.")
            elif add_user(au.strip(), ap, ar):
                st.toast(f"✅ User '{au.strip()}' created", icon="👤"); st.rerun()
            else:
                st.error("Username already exists.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="slabel" style="margin:24px 0 8px">Registered Users</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(get_all_users()), use_container_width=True)

    applications = get_all_applications()
    if not applications:
        st.info("No applications stored yet."); return
    app_df = pd.DataFrame(applications)
    st.markdown('<div class="slabel" style="margin:24px 0 8px">All Applications</div>', unsafe_allow_html=True)
    st.dataframe(app_df, use_container_width=True)
    st.download_button("⬇  Download All as CSV",
        data=app_df.to_csv(index=False).encode("utf-8"),
        file_name="all_applications.csv", mime="text/csv")
    st.markdown('<div class="slabel" style="margin:24px 0 8px">Income vs Loan Regression</div>', unsafe_allow_html=True)
    render_regression_chart(app_df, "income", "loan",
        "Income vs Loan Amount", "Annual Income (₹)", "Loan Amount (₹)")
    st.markdown('<div class="slabel" style="margin:24px 0 8px">Delete Application</div>', unsafe_allow_html=True)
    opts = {f'#{r["id"]} — {r["applicant_name"]} ({r["risk"]})': r["id"] for r in applications}
    sel  = st.selectbox("Select", list(opts.keys()), label_visibility="collapsed")
    if st.button("🗑  Delete Selected"):
        if delete_application(opts[sel]):
            st.toast("🗑 Application deleted", icon="🗑"); st.rerun()
        else:
            st.error("Not found.")


# ══════════════════════════════════════════════════════════════════════════════
# SUPPORT & HELP CENTRE
# ══════════════════════════════════════════════════════════════════════════════
def show_support():
    import uuid as _uuid

    st.markdown('<div class="page-title">Support & Help Centre</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Everything you need — FAQs, bank helplines, guides, and direct support</div>', unsafe_allow_html=True)

    st.html("""
    <div class="sup-hero">
        <div style="font-size:36px;margin-bottom:10px">🛟</div>
        <div style="font-size:22px;font-weight:800;color:#e2e8f0;margin-bottom:6px">How can we help you?</div>
        <div style="font-size:14px;color:#64748b;max-width:520px">
            Browse our knowledge base below, check bank helplines, or submit a support ticket.
            Our team responds within 2–4 business hours on weekdays.
        </div>
    </div>
    """)

    st.html("""
    <div class="sup-kpi-row">
        <div class="sup-kpi"><div class="sup-kpi-val">2–4h</div><div class="sup-kpi-lbl">Avg. Response Time</div></div>
        <div class="sup-kpi"><div class="sup-kpi-val">24/7</div><div class="sup-kpi-lbl">Chat Bot Available</div></div>
        <div class="sup-kpi"><div class="sup-kpi-val">30+</div><div class="sup-kpi-lbl">Help Articles</div></div>
        <div class="sup-kpi"><div class="sup-kpi-val">6</div><div class="sup-kpi-lbl">Bank Helplines</div></div>
    </div>
    """)

    # ── FAQ ───────────────────────────────────────────────────────────────────
    st.markdown('<div class="slabel" style="margin:8px 0 14px">Frequently Asked Questions</div>', unsafe_allow_html=True)

    faqs = [
        ("How is the risk score calculated?",
         "RiskOra uses a two-layer system: (1) A Random Forest ML model trained on 100 decision trees analyzes Age, Income, Loan Amount, Credit Score, and EMI. (2) A rule engine overrides the ML result if: credit score < 550, DTI ratio > 60%, or loan amount > 6× annual income. The final verdict is HIGH RISK or LOW RISK."),
        ("What credit score do I need for a LOW RISK verdict?",
         "A credit score of 650 or above generally leads to a LOW RISK verdict, assuming other factors (DTI, loan amount) are within acceptable ranges. Scores below 550 automatically trigger HIGH RISK regardless of other factors. Scores between 550–649 are borderline — the ML model and other factors determine the final verdict."),
        ("How is DTI (Debt-to-Income) ratio calculated?",
         "DTI = (Monthly EMI × 12 / Annual Income) × 100. For example: if your monthly EMI is ₹20,000 and annual income is ₹8,00,000, DTI = (20,000 × 12 / 8,00,000) × 100 = 30%. A DTI above 60% automatically triggers HIGH RISK."),
        ("Can I appeal a HIGH RISK decision?",
         "RiskOra is a decision-support tool, not a final lending authority. A HIGH RISK verdict means you should review the contributing factors (credit score, DTI, loan-to-income ratio). You can improve your profile by reducing the loan amount, paying off existing debts to lower DTI, or improving your credit score before reapplying."),
        ("How do I use the EMI Calculator?",
         "Navigate to 'EMI Calculator' in the sidebar. Enter: Loan Amount (₹), Annual Interest Rate (%), and Loan Tenure (months). The calculator shows your exact monthly EMI, total payment, total interest, and a full amortization schedule table. You can also see a pie chart showing principal vs interest breakdown."),
        ("What data is stored in the system?",
         "RiskOra stores: applicant name, email, phone, loan purpose, age, income, loan amount, credit score, EMI, DTI, financial score, risk verdict, submitting user, and timestamp. Passwords are stored as PBKDF2-SHA256 hashes — never in plain text. No raw passwords or sensitive financial documents are stored."),
        ("How do I create an admin account?",
         "During signup, select 'admin' in the Account Role dropdown. Existing users can be promoted by an admin via the Admin Panel → Create New User form. Note: admin accounts can view all applications and manage users — assign this role carefully."),
        ("Why is my application not showing in the Dashboard?",
         "Make sure you're logged in with the same account you used to submit the application. Regular users only see their own applications. If you're an admin, check the Admin Panel for the full list. If the application is still missing, it may have been deleted — contact your administrator."),
        ("Is my financial data secure?",
         "Yes. RiskOra uses PBKDF2-SHA256 with 100,000 iterations for password hashing, parameterized SQL queries to prevent injection attacks, and Streamlit's server-side session state (no client-side storage). All data is stored locally in an SQLite database on the server."),
        ("How accurate is the AI model?",
         "The Random Forest model is cross-validated on our dataset. However, it is a decision-support tool trained on synthetic data — not a replacement for professional credit underwriting. Real lending decisions should combine AI insights with human judgment, additional documentation, and regulatory guidelines."),
    ]

    for q, a in faqs:
        with st.expander(f"❓  {q}"):
            st.markdown(f'<div style="font-size:14px;color:#94a3b8;line-height:1.75;padding:4px 0">{a}</div>', unsafe_allow_html=True)

    # ── Bank Helplines ────────────────────────────────────────────────────────
    st.markdown('<div class="slabel" style="margin:28px 0 14px">Bank & Financial Helplines 🇮🇳</div>', unsafe_allow_html=True)

    st.html("""
    <div class="bank-grid">
        <div class="bank-card">
            <div class="bank-name">🏦 State Bank of India</div>
            <div class="bank-num">1800 11 2211</div>
            <div class="bank-sub">Toll-free · 24×7</div>
            <div class="bank-sub" style="margin-top:4px">Alt: 1800 425 3800</div>
        </div>
        <div class="bank-card">
            <div class="bank-name">🏦 HDFC Bank</div>
            <div class="bank-num">1800 202 6161</div>
            <div class="bank-sub">Toll-free · 24×7</div>
            <div class="bank-sub" style="margin-top:4px">Alt: 1800 258 3838</div>
        </div>
        <div class="bank-card">
            <div class="bank-name">🏦 ICICI Bank</div>
            <div class="bank-num">1800 200 3344</div>
            <div class="bank-sub">Toll-free · 24×7</div>
            <div class="bank-sub" style="margin-top:4px">Alt: 1860 120 7777</div>
        </div>
        <div class="bank-card">
            <div class="bank-name">🏦 Axis Bank</div>
            <div class="bank-num">1800 419 5959</div>
            <div class="bank-sub">Toll-free · 24×7</div>
            <div class="bank-sub" style="margin-top:4px">Alt: 1800 209 5577</div>
        </div>
        <div class="bank-card">
            <div class="bank-name">🏦 Kotak Mahindra</div>
            <div class="bank-num">1860 266 2666</div>
            <div class="bank-sub">24×7 Customer Care</div>
            <div class="bank-sub" style="margin-top:4px">WhatsApp: 93222 87777</div>
        </div>
        <div class="bank-card">
            <div class="bank-name">🏛️ RBI Ombudsman</div>
            <div class="bank-num">14448</div>
            <div class="bank-sub">Banking complaints</div>
            <div class="bank-sub" style="margin-top:4px">cms.rbi.org.in</div>
        </div>
    </div>
    """)

    # ── Help Articles ─────────────────────────────────────────────────────────
    st.markdown('<div class="slabel" style="margin:28px 0 14px">Help Articles & Guides</div>', unsafe_allow_html=True)

    articles = [
        ("🎓", "Understanding Credit Risk Scoring", "A complete guide to how AI-based credit risk models work, including feature importance, model bias considerations, and how to interpret HIGH vs LOW risk verdicts in real-world lending.", "Guide", "art-tag-c"),
        ("📐", "How to Improve Your DTI Ratio", "Step-by-step strategies to reduce your Debt-to-Income ratio: consolidating debts, increasing income, prepaying high-interest loans, and using the EMI calculator to plan repayments effectively.", "Financial Tips", "art-tag-g"),
        ("💳", "CIBIL Score — Everything You Need to Know", "What affects your CIBIL score, how to dispute errors in your credit report, how long negative marks stay, and a roadmap from 550 to 800+ in 12 months.", "Credit Guide", "art-tag-p"),
        ("🔐", "Data Security in RiskOra", "Detailed breakdown of RiskOra's security architecture: PBKDF2-SHA256 hashing, salted passwords, parameterized SQL queries, session management, and what data is stored vs discarded.", "Security", "art-tag-c"),
        ("🏠", "Choosing the Right Loan Product", "Comparison of home loans, personal loans, car loans, education loans, and gold loans — interest rates, tenure options, eligibility criteria, and tax benefits under Indian law.", "Loan Guide", "art-tag-g"),
    ]

    for icon, title, desc, tag, tag_cls in articles:
        st.html(f"""
        <div class="art-card">
            <div class="art-icon">{icon}</div>
            <div>
                <div class="art-title">{title}</div>
                <div class="art-desc">{desc}</div>
                <span class="art-tag {tag_cls}">{tag}</span>
            </div>
        </div>
        """)

    # ── Contact Form ──────────────────────────────────────────────────────────
    st.markdown('<div class="slabel" style="margin:28px 0 14px">Submit a Support Ticket</div>', unsafe_allow_html=True)

    with st.form("support_ticket"):
        sc1, sc2 = st.columns(2, gap="large")
        with sc1:
            ct_name  = st.text_input("Full Name", value=st.session_state.get("username",""), placeholder="Your full name")
            ct_email = st.text_input("Email Address", placeholder="your@email.com")
        with sc2:
            st.selectbox("Category", ["Loan Assessment Query","Credit Score Help","Technical Issue","Account & Login","EMI Calculator","Data / Privacy","Other"])
            st.selectbox("Priority", ["Low — General enquiry","Medium — Affects workflow","High — Critical issue"])
        ct_sub   = st.text_input("Subject", placeholder="Brief description of your issue")
        ct_msg   = st.text_area("Message", placeholder="Describe your issue in detail. Include any error messages, application IDs, or steps to reproduce.", height=130)
        submitted = st.form_submit_button("📩  Submit Ticket")

    if submitted:
        if not ct_name.strip() or not ct_email.strip() or not ct_sub.strip() or not ct_msg.strip():
            st.error("Please fill in all fields before submitting.")
        elif "@" not in ct_email or "." not in ct_email:
            st.error("Please enter a valid email address.")
        else:
            ticket_id = "RO-" + str(_uuid.uuid4())[:8].upper()
            st.success(f"✅ Ticket submitted successfully! Your reference: **{ticket_id}**")
            st.info(f"We'll respond to **{ct_email}** within 2–4 business hours (Mon–Fri, 9 AM – 6 PM IST).")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
_inject_css()

if not st.session_state.logged_in:
    if st.session_state.get("page", "landing") == "landing":
        show_landing()
    else:
        show_auth()
else:
    role_cls = "sb-role-admin" if st.session_state.role == "admin" else "sb-role-user"
    st.sidebar.html(f"""
    <div style="padding:20px 4px 16px">
        <div class="sb-logo">{APP_SHORT}</div>
        <div class="sb-logo-sub">AI Risk Intelligence</div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(147,51,234,.3),transparent);margin-bottom:14px"></div>
    <div class="slabel" style="padding:0 4px">User Panel</div>
    <div class="sb-user">
        <div style="font-size:24px;margin-bottom:6px">👤</div>
        <div class="sb-username">{st.session_state.username}</div>
        <span class="{role_cls}">{st.session_state.role}</span>
    </div>
    """)

    if st.sidebar.button("⏻  Logout"):
        for k in ["logged_in","role","username","data","chat_history","auth_tab","page"]:
            st.session_state[k] = (False if k == "logged_in" else
                                   ([] if k == "chat_history" else
                                    ("login" if k == "auth_tab" else
                                     ("landing" if k == "page" else None))))
        st.rerun()

    st.sidebar.markdown('<div class="sb-nav-label">Navigation</div>', unsafe_allow_html=True)
    menu = ["🏠  Home","📋  Apply Loan","🧮  EMI Calculator","📊  Dashboard","🔮  AI Assistant","📞  Support"]
    if st.session_state.role == "admin":
        menu.append("🛡️  Admin Panel")
    choice = st.sidebar.selectbox("", menu)

    if   "Home"        in choice: show_home()
    elif "Apply Loan"  in choice: show_apply_loan()
    elif "EMI"         in choice: show_emi_calculator()
    elif "Dashboard"   in choice: show_dashboard()
    elif "Assistant"   in choice: show_chatbot()
    elif "Support"     in choice: show_support()
    elif "Admin Panel" in choice: show_admin_panel()

    show_chat_widget()
