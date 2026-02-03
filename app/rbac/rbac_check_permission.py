from app.auth.auth import UserInDB
from app.rbac.perm import check_permission


class KBPermissions:
    KB_MANAGE_DOCS = "kb.manage_docs"


class LeavePermission:
    LEAVE_APPROVE = "leave.approve"


def require_kb_manage_docs(user: UserInDB) -> None:
    check_permission(user, KBPermissions.KB_MANAGE_DOCS)
