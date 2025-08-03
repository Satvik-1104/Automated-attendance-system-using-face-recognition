import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "../uploads")

ENTRY_LOGS_DIR = os.path.join(UPLOADS_DIR, "entry_logs")
EXIT_LOGS_DIR = os.path.join(UPLOADS_DIR, "exit_logs")
PROCESSED_ENTRY_LOGS_DIR = os.path.join(UPLOADS_DIR, "processed_entry_logs")
PROCESSED_EXIT_LOGS_DIR = os.path.join(UPLOADS_DIR, "processed_exit_logs")

COURSE_GROUPS = {
    2022: {
        "core_courses": {
            "CSE": [
                {"course_code": "CS361", "sections": ["CS31", "CS32"]},
                {"course_code": "CS320", "sections": ["CS31", "CS32"]},
                {"course_code": "CS330", "sections": ["CS31", "CS32"]},
                {"course_code": "CS321", "sections": ["CS31", "CS32"]},
                {"course_code": "CS331", "sections": ["CS31", "CS32"]}
            ],
            "ECE": [
                # {"course_code": "EC353", "sections": ["EC31"]},
                {"course_code": "EC361", "sections": ["EC3"]},
                # {"course_code": "EC371", "sections": ["EC31"]},
                # {"course_code": "EC381", "sections": ["EC31"]},
                {"course_code": "EC372", "sections": ["EC31", "EC32"]},
                # {"course_code": "EC382", "sections": ["EC31", "EC32"]}
            ]
        },
        "electives_cs": [
            {"course_code": "CS481", "sections": ["CS481"]},
            {"course_code": "CS653", "sections": ["CS653"]},
            {"course_code": "CS300", "sections": ["CS300"]}
        ],
        "electives_ec": [
            {"course_code": "EC300", "sections": ["EC300"]},
            {"course_code": "None", "sections": ["None"]}
        ],
        "humanities": [
            {"course_code": "HS308", "sections": ["HS308"]},
            {"course_code": "HS307", "sections": ["HS307"]}
        ],
        "science": [
            {"course_code": "SC302", "sections": ["S31", "S32", "S33"]}
        ]
    }
}
