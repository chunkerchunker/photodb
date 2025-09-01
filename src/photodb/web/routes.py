from flask import Blueprint, render_template, abort, request
from .queries import PhotoQueries
from .images import serve_image
import json
import calendar
import os

bp = Blueprint("main", __name__)
queries = PhotoQueries()


@bp.route("/")
def index():
    years = queries.get_years_with_photos()
    return render_template("index.html", years=years)


@bp.route("/year/<int:year>")
def year_view(year):
    months = queries.get_months_in_year(year)

    month_data = []
    for month_info in months:
        month_data.append(
            {
                "month": month_info["month"],
                "month_name": calendar.month_name[month_info["month"]],
                "photo_count": month_info["photo_count"],
                "sample_photo_ids": month_info.get("sample_photo_ids", [])[:4],
            }
        )

    return render_template("year.html", year=year, months=month_data)


@bp.route("/year/<int:year>/month/<int:month>")
def month_view(year, month):
    page = request.args.get("page", 1, type=int)
    per_page = 48  # 48 photos per page for 6 columns x 8 rows
    offset = (page - 1) * per_page

    photos = queries.get_photos_by_month(year, month, limit=per_page, offset=offset)
    total_photos = queries.get_photo_count_by_month(year, month)
    total_pages = (total_photos + per_page - 1) // per_page

    for photo in photos:
        photo["filename_only"] = os.path.basename(photo["filename"])
        if photo.get("description"):
            photo["short_description"] = (
                (photo["description"][:47] + "...")
                if len(photo["description"]) > 50
                else photo["description"]
            )
        else:
            photo["short_description"] = None

    month_name = calendar.month_name[month]

    return render_template(
        "month.html",
        year=year,
        month=month,
        month_name=month_name,
        photos=photos,
        page=page,
        total_pages=total_pages,
        total_photos=total_photos,
    )


@bp.route("/photo/<photo_id>")
def photo_detail(photo_id):
    photo = queries.get_photo_details(photo_id)
    if not photo:
        abort(404, f"Photo {photo_id} not found")

    photo["filename_only"] = os.path.basename(photo["filename"])

    if photo.get("analysis"):
        try:
            if isinstance(photo["analysis"], str):
                photo["analysis_formatted"] = json.dumps(json.loads(photo["analysis"]), indent=2)
            else:
                photo["analysis_formatted"] = json.dumps(photo["analysis"], indent=2)
        except:
            photo["analysis_formatted"] = str(photo.get("analysis", ""))

    if photo.get("metadata_extra"):
        try:
            if isinstance(photo["metadata_extra"], str):
                photo["metadata_formatted"] = json.dumps(
                    json.loads(photo["metadata_extra"]), indent=2
                )
            else:
                photo["metadata_formatted"] = json.dumps(photo["metadata_extra"], indent=2)
        except:
            photo["metadata_formatted"] = str(photo.get("metadata_extra", ""))

    if photo.get("captured_at"):
        year = photo["captured_at"].year
        month = photo["captured_at"].month
        photo["year"] = year
        photo["month"] = month
        photo["month_name"] = calendar.month_name[month]

    return render_template("photo.html", photo=photo)


@bp.route("/api/image/<photo_id>")
def serve_photo(photo_id):
    photo = queries.get_photo_by_id(photo_id)
    if not photo:
        abort(404, f"Photo {photo_id} not found")

    return serve_image(photo["normalized_path"])
