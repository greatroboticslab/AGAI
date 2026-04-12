# UI Components for Plant Diagnostic System

SIDEBAR_HEADER = """
<div style="text-align: center; padding: 20px 0 10px 0;">
    <h1 style="font-size: 1.8rem; margin: 0; background: linear-gradient(135deg, #10a37f, #14d49a); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Plant Diagnostic</h1>
    <p style="color: #888; font-size: 0.9rem; margin-top: 4px;">Dual Architecture Analysis</p>
</div>
"""

CHAT_FOOTER = """
<div style="text-align: center; padding: 8px 0; color: #666; font-size: 12px;">
    GreatRoboticsLab @ MTSU
</div>
"""

ABOUT_SECTION = """
<div style="max-width: 720px; margin: 0 auto; padding: 50px 28px; animation: fadeIn 0.6s ease;">
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        .team-card {
            display: flex;
            align-items: center;
            gap: 18px;
            padding: 16px;
            border-radius: 12px;
            transition: all 0.3s ease;
            animation: slideIn 0.5s ease backwards;
        }
        .team-card:hover {
            background: rgba(16, 163, 127, 0.08);
            transform: translateX(8px);
        }
        .team-card:nth-child(1) { animation-delay: 0.1s; }
        .team-card:nth-child(2) { animation-delay: 0.2s; }
        .team-card:nth-child(3) { animation-delay: 0.3s; }
        .avatar {
            width: 52px;
            height: 52px;
            background: #2a2a2a;
            border: 1px solid #3a3a3a;
            border-radius: 14px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #b0b0b0;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        .team-card:hover .avatar {
            border-color: #10a37f;
            box-shadow: 0 0 20px rgba(16, 163, 127, 0.3);
        }
        .info-card {
            background: #1a1a1a;
            border: 1px solid #2a2a2a;
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 24px;
            transition: all 0.3s ease;
        }
        .info-card:hover {
            border-color: rgba(16, 163, 127, 0.4);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 20px rgba(16, 163, 127, 0.1);
        }
    </style>

    <h2 style="font-size: 2.2rem; font-weight: 700; margin-bottom: 8px; color: #ffffff; animation: fadeIn 0.5s ease;">Plant Diagnostic System</h2>
    <p style="color: #888; font-size: 1.1rem; margin-bottom: 45px; animation: fadeIn 0.6s ease 0.1s backwards;">Advanced dual-architecture plant disease analysis</p>
    
    <div class="info-card">
        <h3 style="color: #10a37f; font-size: 0.95rem; font-weight: 600; margin-bottom: 24px; text-transform: uppercase; letter-spacing: 1px;">Research Team</h3>
        
        <div style="display: grid; gap: 8px;">
            <div class="team-card">
                <div class="avatar">WS</div>
                <div>
                    <p style="color: #ffffff; font-weight: 600; margin: 0; font-size: 1.05rem;">William Starks</p>
                    <p style="color: #888; font-size: 0.9rem; margin: 0;">Lead Developer</p>
                </div>
            </div>
            
            <div class="team-card">
                <div class="avatar">GM</div>
                <div>
                    <p style="color: #ffffff; font-weight: 600; margin: 0; font-size: 1.05rem;">Gus Marcum</p>
                    <p style="color: #888; font-size: 0.9rem; margin: 0;">Developer</p>
                </div>
            </div>
            
            <div class="team-card">
                <div class="avatar">HZ</div>
                <div>
                    <p style="color: #ffffff; font-weight: 600; margin: 0; font-size: 1.05rem;">Dr. Hongbo Zhang</p>
                    <p style="color: #888; font-size: 0.9rem; margin: 0;">Faculty Advisor</p>
                </div>
            </div>
        </div>
    </div>
    
    <div class="info-card">
        <h3 style="color: #10a37f; font-size: 0.95rem; font-weight: 600; margin-bottom: 18px; text-transform: uppercase; letter-spacing: 1px;">Architecture</h3>
        <p style="color: #e0e0e0; line-height: 1.8; margin: 0; font-size: 1rem;">
            This system combines <span style="color: #14d49a; font-weight: 600;">ResNet-50</span> for rapid disease classification with 
            <span style="color: #14d49a; font-weight: 600;">RF-DETR</span> for plant part detection and <span style="color: #14d49a; font-weight: 600;">MiniGPT-v2</span> for detailed visual analysis. The dual-architecture 
            approach enables both quick identification and comprehensive diagnostic explanations.
        </p>
    </div>
    
    <p style="text-align: center; color: #555; font-size: 0.85rem; margin-top: 40px;">
        GreatRoboticsLab · Middle Tennessee State University
    </p>
</div>
"""
