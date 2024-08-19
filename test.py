from flask import Flask, session, redirect, url_for, request, render_template

app = Flask(__name__)

# 设置一个密钥来加密 session 数据
app.secret_key = 'supersecretkey'

@app.route('/')
def index():
    # 获取 session 中的 'username'，如果不存在则使用 'Guest' 作为默认值
    username = session.get('username', 'Guest')
    breakpoint()
    return f'Hello, {username}! <br><a href="/login">Login</a> | <a href="/logout">Logout</a>'

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # 从表单中获取用户名并存储到 session 中
        session['username'] = request.form['username']
        return redirect(url_for('index'))
    return '''
        <form method="post">
            Username: <input type="text" name="username">
            <input type="submit" value="Login">
        </form>
    '''

@app.route('/logout')
def logout():
    # 删除 session 中的 'username'
    session.pop('username', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
