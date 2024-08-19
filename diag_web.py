from flask import Flask, render_template, request, redirect, url_for, flash, session
from utils import load_user_data, save_user_data
import os
import math
import json
import matplotlib.pyplot as plt
import matplotlib
from collections import OrderedDict
from io import BytesIO
import base64
from matplotlib.figure import Figure
import numpy as np
import pickle
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from flask_session import Session

matplotlib.use('Agg')
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 必须设置用于 session 和 flash 消息
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
font_manager.fontManager.addfont(os.path.join('static', 'SimHei.ttf'))

def load_translations(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        return json.load(file)
def get_translation(translations, lang, key):
    return translations.get(lang, {}).get(key, key)
def get_text(key):
    lang = session.get('lang', 'English')
    return get_translation(session.get('translations'), lang, key)

# Helper function to validate string input
def validate_str(_str):
    return '0' if (_str == '-' or _str == '' or _str == ' ') else _str

@app.route('/')
def landing():
    # 加载用户数据
    session['user_data'] = load_user_data(os.path.join("static", "user_data.json"))
    session['translations'] = load_translations(os.path.join("static", "translations.json"))
    session['score_save_flag'] = False
    return render_template('page1.html', title=get_text("title"),
                               login_text=get_text("Register/Login"))
        
        
@app.route('/home')
def home():
    return render_template('page1s.html', title=get_text("title"), 
                               Begindetection=get_text("begin_detection"),
                               Logout=get_text("Logout"))
    
@app.route('/set_language', methods=['POST'])
def set_language():
    session['lang'] = request.form.get('lang', 'English')
    return redirect(url_for('landing'))

@app.route('/logout')
def logout():
    # 将 current_user 设置为空字符串
    session['current_user'] = ''
    # 重定向到主页
    return redirect(url_for('landing'))

@app.route('/pagelogin', methods=['GET', 'POST'])
def page_login():
    if request.method == 'POST':
        action = request.form['action']  # 判断用户点击了哪个按钮
        username = request.form['username']
        password = request.form['password']
        user_data = session.get('user_data')
        if action == 'Login':
            if username in user_data and user_data[username] == password:
                session['current_user'] = username
                return redirect(url_for('home'))
            else:
                flash(get_text('Invalid username or password'), 'danger')
                return redirect(url_for('page_login'))

        elif action == 'Register':
            if username in user_data:
                flash(get_text('Username already exists'), 'danger')
            elif username.strip() == '' or password.strip() == '':
                flash(get_text('Invalid username or password: cannot be empty'), 'danger')
            else:
                user_data[username] = password
                flash(get_text('Registration successful, please login'), 'success')
                save_user_data(os.path.join("static", "user_data.json"), user_data)
                return redirect(url_for('page_login'))

    return render_template('pagelogin.html', Welcome = get_text("Welcome"), 
                           Login = get_text("Login"),
                           Register = get_text("Register"),
                           Username = get_text("Username:"),
                           Password = get_text("Password:"))

# The Page2 view
@app.route('/page2', methods=['GET', 'POST'])
def page2():
    current_user = session.get('current_user','')
    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'submit':
            labels = request.form.to_dict()
            save_suggestions(current_user, labels)
            for i, field in enumerate(request.form.keys()):
                if request.form[field].strip() == '' and i < 28:
                    flash(get_text('Please fill in all required fields.'), 'danger')
                    return redirect(url_for('page2'))
            labels = request.form.to_dict()
            save_suggestions(current_user, labels)
            session['output_labels'] = label_processing(labels)
            session['score_save_flag'] = True
            return redirect(url_for('page3'))
        
        elif action == 'save':
            labels = request.form.to_dict()
            save_suggestions(current_user, labels)
            flash(get_text('Data saved successfully'), 'success')
            return redirect(url_for('page2'))
        
        elif action == 'reset':
            reset_suggestions(current_user)
            flash(get_text('Data reset successfully'), 'success')
            return redirect(url_for('page2'))
        elif action == 'return':
            return redirect(url_for('home'))

    # Load suggestions to pre-fill the form
    suggestions = load_suggestions(current_user)
    labels = [
    "Your age (years)", "Time Lapse (years)", "Body weight (Kg)", "Height (cm)",
    "Limited shoulder movement", "Limited elbow movement", "Limited wrist movement",
    "Limited fingers movement", "Limited arm movement", "Arm or hand swelling",
    "Breast swelling", "Chest swelling", "Toughness or thickness of skin",
    "Pain, aching, soreness", "Tightness", "Heaviness", "Burning", "Tingling",
    "Weakness", "Hotness", "Tenderness", "Firmness", "Numbness", "Stabbing",
    "Fatigue", "Redness", "Stiffness", "Blister", "Chemotherapy", "Radiation",
    "Mastectomy", "Lumpectomy", "Hormonal therapy", 
    "Sentinel lymph node biopsy removal number", "Axillary lymph node dissection number"
    ]
    notations = ["None", "A little", "Somewhat", "Quite a bit", "Severe", "Yes", "No"]
    for i, item in enumerate(labels):
        labels[i] = get_text(item)
    for i, item in enumerate(notations):
        notations[i] = get_text(item)
    return render_template('page2.html', suggestions=suggestions, labels = labels, notations = notations,
                           Reset = get_text("Reset"),
                           Submit = get_text("Submit"),
                           Save = get_text("Save"),
                           Return_Home = get_text("Return/Back Home"),
                           detection_page = get_text("detection_page"))
    
@app.route('/page3')
def page3():
    output_labels = session.get('output_labels', {})
    current_user = session.get('current_user','')
    lang = session.get('lang')
    select_mask = ['Mobility', 'ArmSwelling', 'BreastSwelling', 'Skin', 'FHT', 'DISCOMFORT'\
        , 'SYM_COUNT', 'ChestWallSwelling', 'Mastectomy', 'Lumpectomy', 'TIME_LAPSE']
    data_select = np.array([[output_labels[item] for item in select_mask]], dtype=float)
    model_path = os.path.join('models' , 'GBT.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict_proba(data_select).squeeze()
    max_label = np.argmax(y_pred)
    overall_score = cal_overall_score(y_pred)
    save_score(overall_score)

    # draw
    plt.figure(num = 1, figsize=(8, 3.3), dpi=100)

    # 设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    if lang == 'Chinese':
        font = {'family': 'SimHei', 'weight': 'normal', 'size': 26}
    else:
        font = {'family': 'serif', 'serif': 'Times New Roman', 'weight': 'normal', 'size': 26}
    plt.rc('font', **font)

    # 创建分段渐变条
    plt.title(get_text('Lymphedema score'), pad=60)
    gradient = np.linspace(0, 1, 1000).reshape(1, -1)
    plt.imshow(gradient, aspect='auto', cmap='RdYlGn_r', extent=[0, 100, 0, 1])

    # 绘制风险评分的指示线
    plt.axvline(overall_score, color='black', linewidth=2)
    plt.text(overall_score, 0.5, f'{overall_score:.1f}', color='black', va='center', ha='center', bbox=dict(facecolor='white',edgecolor='none', alpha=0.7))

    # 添加分界线
    threshold1 = 33.3
    threshold2 = 66.7
    plt.axvline(threshold1, color='black', linestyle='--', linewidth=2)
    plt.axvline(threshold2, color='black', linestyle='--', linewidth=2)

    # 添加箭头符号指示当前值，并将箭头放在图像上方
    re_dict = {0: "low_risk", 1: "mild", 2: "moderate_severe"}
    plt.gca().annotate(
        get_text(re_dict[max_label]), xy=(overall_score, 1), xytext=(overall_score, 1.2),
        arrowprops=dict(facecolor='black', shrink=0.1, headwidth=10, width=3),
        ha='center', va='bottom', backgroundcolor='white',
        fontsize=30
    )

    # 设置图形标题和标签
    plt.gca().set_yticks([])
    plt.gca().set_xlim(0, 100)
    plt.tight_layout()

    # 将图像保存为 base64 编码的字符串
    output = BytesIO()
    plt.savefig(output, format='png')
    encoded_img = base64.b64encode(output.getvalue()).decode('utf-8')
    plt.close()
    
    # 将图像数据嵌入到 HTML 中
    plot_frame_content = f'<img src="data:image/png;base64,{encoded_img}"/>'
    
    # Generate comments and suggestions based on logic
    with open(os.path.join("static", "user_record.json"), 'r') as json_file:
        existing_data = json.load(json_file)
        try:
            score_list = existing_data[current_user]['score_list']
        except:
            score_list = []
    comments = []
    suggestions = []

    if len(score_list) == 0 or len(score_list) == 1:
        if max_label == 1 or max_label == 2:
            comments.append(get_text('This is your first time detecting Lymphedema. Keep on excercising and let us see your progress!'))
        elif max_label == 0:
            comments.append(get_text('This is your first time detecting Lymphedema. Keep on the good record!'))
    else:
        last_time_score = score_list[-2]
        if max_label == 0:
            comments.append(get_text('Your detection result shows low risk, keep on the good record!'))
        elif overall_score > last_time_score and (max_label == 1 or max_label == 2):
            comments.append(get_text('Your detection result requires further inspection, please advice the doctors for further help. Keep on excercising and let us see your progress!'))
        elif overall_score == last_time_score and (max_label == 1 or max_label == 2):
            comments.append(get_text('Your detection result does not change since the last time. Keep on excercising and let us see your progress!'))
        elif overall_score < last_time_score - 3 and (max_label == 1 or max_label == 2):
            comments.append(get_text('Congratulations! Your detection result is much better than your last time. Keep on the good record!'))
        elif overall_score < last_time_score and (max_label == 1 or max_label == 2):
            comments.append(get_text('Congratulations! Your detection result is better than your last time. Keep on excercising and keep on the good record!'))

    suggestions.append(get_text(f'diag_{re_dict[max_label]}'))
    
    return render_template('page3.html', title = get_text("Visualized Diagnosis"), plot_frame_content=plot_frame_content, comments=comments, suggestions=suggestions, 
                           ct = get_text('Comments'), st = get_text('Suggestions'),
                           History = get_text('History'), Risk_Analysis = get_text('Risk Analysis'), Return = get_text('return'), Back_Home = get_text('Back Home'))

@app.route('/pagehistory', methods=['GET'])
def pagehistory():
    lang = session.get('lang', 'English')
    current_user = session.get('current_user', '')
    # 从 JSON 文件中读取用户的得分历史
    with open(os.path.join("static", "user_record.json"), 'r') as json_file:
        existing_data = json.load(json_file)
        score_list = existing_data[current_user]['score_list']
    
    # 获取用户的选择
    show = request.args.get('show', '5')  # 默认显示最近 5 次

    # 选择要显示的得分数量
    if show.isdigit():
        num_scores = int(show)
        
        score_list = score_list[-num_scores:]
    elif show == 'all':
        num_scores = len(score_list)
        score_list = score_list[-num_scores:]

    # 绘制图表
    length = min(max(num_scores, 8), 20)
    # 设置标题和标签
    plt.figure(figsize=(length, 6.8), dpi=100)
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    if lang == 'Chinese':
        font = {'family': 'SimHei', 'weight': 'normal', 'size': 26}
    else:
        font = {'family': 'serif', 'serif': 'Times New Roman', 'weight': 'normal', 'size': 26}
    plt.rc('font', **font)
    
    plt.title(get_text('Lymphedema Score History'), fontsize=32, pad=20)
    plt.xlabel(get_text('Test Number'), fontsize=25 )
    plt.ylabel(get_text('Score'), fontsize=25 )

    def plot_scores(scores):
        x = range(1, len(scores) + 1)
        plt.plot(x, scores, marker='o', linestyle='-', color='b', linewidth=3 , label='Scores')
        plt.ylim(min(scores) - 5, max(scores) + 5)
        plt.xticks(np.arange(1, len(scores) + 1, 1))
        for i, score in enumerate(scores):
            plt.text(x[i], score + 0.4, f'{score:.1f}', fontsize=18, ha='center')
        plt.grid(True)

    # 显示选择的得分数据
    plot_scores(score_list)
    
    # 保存图表为图片
    fig_path = os.path.join("static", "score_history.png")
    plt.tight_layout()
    plt.savefig(fig_path)

    return render_template('pagehistory.html', title=get_text("Detection History"), fig_path=fig_path, length=length,
                           l5 = get_text('last 5 times'),
                           l10 = get_text('last 10 times'),
                           l20 = get_text('last 20 times'),
                           ov = get_text('Overall'),
                           Return = get_text('return'),
                           Back_Home = get_text('Back Home'),)


@app.route('/pagefactor')
def pagefactor():
    # 调用绘图函数生成图像
    img = create_figure_factor()

    # 将生成的图像编码为 base64
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # 渲染模板并传递图像数据
    return render_template('pagefactor.html', title=get_text("Risk Analysis"), plot_url=plot_url,
                           Return = get_text('return'),
                           Back_Home = get_text('Back Home'))

def create_figure_factor():
    lang = session.get('lang', 'English')
    output_labels = session.get('output_labels', {})
    plt.figure(num=4, figsize=(16, 10), dpi=80, facecolor="white", edgecolor='Teal', frameon=True)
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    if lang == 'Chinese':
        font = {'family': 'SimHei', 'weight': 'normal', 'size': 26}
    else:
        font = {'family': 'serif', 'serif': 'Times New Roman', 'weight': 'normal', 'size': 26}
    plt.rc('font', **font)
    plt.title(get_text('Your Risk Factors Contributing to Lymphedema'), pad=20, fontsize=36)
    plt.xscale('symlog', linthresh=0.00005)
    plt.tick_params(axis='y', labelsize=18)
    font_prop = FontProperties(family='Times New Roman')
    plt.tick_params(axis='x', labelsize=18)
    for label in plt.gca().get_xticklabels():
        label.set_fontproperties(font_prop)
    
    data = []
    if int(output_labels['ArmSwelling']) > 0:
        data.append(('Arm or hand swelling', 0.5666504441537034))
    if output_labels['SYM_COUNT'] > 5:
        data.append(('Related symptom count', 0.32829106757634513))
    if int(output_labels['BreastSwelling']) > 0:
        data.append(('Breast swelling', 0.05866949630997336))
    if output_labels['BMI'] > 23.5 or output_labels['BMI'] < 19:
        data.append(('Height and weight (BMI)', 0.0048442873079247665))
    if int(output_labels['FHT']) > 0:
        data.append(('Tightness, firmness, and heaviness', 0.003799860520333767))
    if int(output_labels['Skin']) > 0:
        data.append(('Toughness of skin', 0.0018357913101027619))
    if int(output_labels['DISCOMFORT']) > 0:
        data.append(('Discomfort', 0.0009075648872365948))
    if int(output_labels['PAS']) > 0:
        data.append(('Pain, aching and soreness', 0.0007936402703285446))
    if int(output_labels['Mobility']) > 0:
        data.append(('Limited Mobility', 0.000173412474200293))
    if int(output_labels['ChestWallSwelling']) > 0:
        data.append(('ChestWall swelling', 0.0001060488637141992))
    
    x = [get_text(item[0]) for item in data]
    y = [item[1] for item in data]
    loc = zip(x, y)
    plt.xlim(0, 1)
    plt.barh(x, y, facecolor='Teal')
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.gca().invert_yaxis()  # 反转纵轴顺序
    for x, y in loc:
        if y >= 0.0001:
            plt.text((1+0.1)*y, x, '%.1g' % y, va='center')
        else:
            plt.text((1+0.1)*y, x, '<0.0001', va='center')

    plt.tight_layout()
    # plt.subplots_adjust(left=0.255, right=0.9, top=0.91, bottom=0.01)

    # 将图像保存为 base64 编码的字符串
    output = BytesIO()
    plt.savefig(output, format='png')
    plt.close()
    
    return output


# Save suggestions function
def save_suggestions(current_user, suggestions):
    existing_data = load_user_data(os.path.join("static", "user_record.json"))
    try:
        existing_data[current_user]['suggestions'] = suggestions
    except:
        existing_data[current_user] = {}
        existing_data[current_user]['suggestions'] = suggestions
    with open(os.path.join("static", "user_record.json"), 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

# Load suggestions function
def load_suggestions(user):
    if os.path.exists(os.path.join("static", "user_record.json")):
        data = load_user_data(os.path.join("static", "user_record.json"))
        if user in data:
            return data[user].get('suggestions', {})
    return {}

# Reset suggestions function
def reset_suggestions(user):
    existing_data = load_user_data(os.path.join("static", "user_record.json"))
    if user in existing_data:
        existing_data[user]['suggestions'] = {}
    save_user_data(os.path.join("static", "user_record.json"), existing_data)

def cal_overall_score(y_pred):
    max_index = np.argsort(y_pred)[-1]
    submax_index = np.argsort(y_pred)[-2]
    if max_index == 2:
        base_score = 2 / 3 
        bias = (y_pred[max_index] - y_pred[submax_index]) / 3
        overall_score = base_score + bias
    elif max_index == 0:
        base_score = 0 / 3 
        bias = (y_pred[max_index] - y_pred[submax_index]) / 3
        overall_score = base_score + bias
    elif max_index == 1:
        base_score = 1 / 2 
        bias = (y_pred[max_index] - y_pred[submax_index]) / 6
        if submax_index == 2:
            overall_score = base_score + bias
        else:
            overall_score = base_score - bias
    overall_score *= 100
    return overall_score

# Label processing function (converted from the original method)
def label_processing(labels):
    def str2b(str_):
        return 0 if str_ == '0' else 1
    output_labels = OrderedDict({
        'BMI': "22.1", 'Age': "40", 'TIME_LAPSE': "1", 'Mobility': "1", 
        'ArmSwelling': "0", 'BreastSwelling': "0", 'Skin': "0", 
        'PAS': "0", 'FHT': "1", 'DISCOMFORT': "0", 'SYM_COUNT': "2", 
        'ChestWallSwelling': "0", 'Chemotherapy': "1", 'Radiation': "0", 
        'Number_nodes': "1", 'Mastectomy': "1", 'Lumpectomy': "0", 'Hormonal': "0"
    })
    output_labels['BMI'] = float(labels['Body weight (Kg)']) / ((float(labels['Height (cm)'])/100)**2)
    output_labels['Age'] = labels['Age (years)']
    output_labels['TIME_LAPSE'] = math.log(float(labels['Time Lapse (years)']))
    output_labels['Mobility'] = max(
        int(labels['Limited shoulder movement']), 
        int(labels['Limited elbow movement']), 
        int(labels['Limited wrist movement']), 
        int(labels['Limited fingers movement']), 
        int(labels['Limited arm movement'])
    )
    output_labels['ArmSwelling'] = labels['Arm or hand swelling']
    output_labels['BreastSwelling'] = labels['Breast swelling']
    output_labels['Skin'] = labels['Toughness or thickness of skin']
    output_labels['PAS'] = labels['Pain, aching, soreness']
    output_labels['FHT'] = max(int(labels['Firmness']), int(labels['Heaviness']), int(labels['Tightness']))
    output_labels['DISCOMFORT'] = labels['Pain, aching, soreness']
    output_labels['SYM_COUNT'] = sum(
        str2b(labels[k]) for k in [
            'Limited shoulder movement', 'Limited elbow movement', 
            'Limited wrist movement', 'Limited fingers movement', 
            'Limited arm movement', 'Arm or hand swelling', 
            'Breast swelling', 'Chest swelling', 
            'Toughness or thickness of skin', 'Pain, aching, soreness', 
            'Tightness', 'Firmness', 'Heaviness', 'Numbness', 
            'Burning', 'Stabbing', 'Tingling', 'Fatigue', 
            'Weakness', 'Redness', 'Hotness', 'Stiffness', 
            'Tenderness', 'Blister'
        ]
    )
    output_labels['ChestWallSwelling'] = labels['Chest swelling']
    output_labels['Chemotherapy'] = validate_str(labels['Chemotherapy'])
    output_labels['Radiation'] = validate_str(labels['Radiation'])
    output_labels['Number_nodes'] = int(validate_str(labels['SLNB_Removed_LN'])) + int(validate_str(labels['ALND_Removed_LN']))
    output_labels['Mastectomy'] = validate_str(labels['Mastectomy'])
    output_labels['Lumpectomy'] = validate_str(labels['Lumpectomy'])
    output_labels['Hormonal'] = validate_str(labels['Hormonal therapy'])
    return output_labels

def save_score(overall_score):
    score_save_flag = session.get('score_save_flag','False')
    current_user = session.get('current_user','')
    if score_save_flag:
        with open(os.path.join("static", "user_record.json"), 'r') as json_file:
            existing_data = json.load(json_file)
            user_dict = existing_data[current_user]
            if 'score_list' not in user_dict:
                user_dict['score_list'] = [overall_score]
            else: 
                user_dict['score_list'].append(overall_score)
        with open(os.path.join("static", "user_record.json"), 'w') as json_file:
            json.dump(existing_data, json_file, indent=4)
        session['score_save_flag'] = False

if __name__ == '__main__':
    app.run(debug=True)
