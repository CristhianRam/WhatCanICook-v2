from flask import Blueprint, redirect, render_template, request, url_for

import db
import services

bp = Blueprint("operations", __name__, template_folder="templates/tasks")


def requires_db(func):
    def wrapper(*args, **kwargs):
        db_context = db.get_db()
        db_cursor = db_context.cursor()
        res = func(db_cursor, *args, **kwargs)
        return res

    return wrapper


@bp.get("/")
def show_home():
    return render_template("tasks/index.html")


@bp.route("/search", methods=["GET", "POST"])
@requires_db
def search(cursor):
    if request.method == "POST":
        q = request.form.get("query", "").strip()
    else:
        q = request.args.get("query", "").strip()

    if not q:
        return redirect(url_for("operations.show_home"))

    results = services.get_recipes(cursor, q, k=10)
    return render_template("tasks/results.html", query=q, results=results)
