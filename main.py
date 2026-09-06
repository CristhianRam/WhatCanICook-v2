import views
from app_init import create_app

app = create_app(views)

if __name__ == "__main__":
    app.run(debug=True)
