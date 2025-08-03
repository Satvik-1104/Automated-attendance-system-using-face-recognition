from sqlalchemy.orm import Session
from sqlalchemy import and_
from ..models import StudentSection, Section
from ..config import COURSE_GROUPS


def validate_selection(batch: int, branch: str, selected: list) -> list:
    errors = []
    config = COURSE_GROUPS.get(batch, {}).get("core_courses", {}).get(branch, [])

    for course in config:
        selected_for_course = [s for s in selected if s in course["sections"]]
        if not selected_for_course:
            errors.append(f"Missing selection for {course['course_code']}")
        elif len(selected_for_course) > 1:
            errors.append(f"Multiple selections for {course['course_code']}")

    return errors


def assign_sections(db: Session, roll_number: int, batch: int, branch: str, selected: list) -> list:
    if errors := validate_selection(batch, branch, selected):
        raise ValueError(" | ".join(errors))

    assigned = []
    config = COURSE_GROUPS.get(batch, {})

    try:
        core_courses = config.get("core_courses", {}).get(branch, [])
        for course in core_courses:
            section_name = next((s for s in selected if s in course["sections"]), None)
            section = db.query(Section).filter(
                and_(
                    Section.section_name == section_name,
                    Section.batch == batch,
                    Section.course_code == course["course_code"]
                )
            ).first()

            if section:
                assigned.append(section.section_id)
                db.add(StudentSection(
                    roll_number=roll_number,
                    section_id=section.section_id
                ))

        for group in ["electives_cs", "electives_ec", "humanities", "science"]:
            for course in config.get(group, []):
                if "None" in course["sections"] and "None" in selected:
                    continue

                section_name = next((s for s in selected if s in course["sections"]), None)
                if section_name:
                    section = db.query(Section).filter(
                        and_(
                            Section.section_name == section_name,
                            Section.batch == batch,
                            Section.course_code == course["course_code"]
                        )
                    ).first()
                    if section:
                        assigned.append(section.section_id)
                        db.add(StudentSection(
                            roll_number=roll_number,
                            section_id=section.section_id
                        ))

        return assigned

    except Exception as e:
        db.rollback()
        raise ValueError(f"Section assignment failed: {str(e)}")
