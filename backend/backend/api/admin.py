from django.contrib import admin

from .models import Academy


@admin.register(Academy)
class AcademyAdmin(admin.ModelAdmin):
    # ISSUE-007 / FR-AUTH-003: this list is the admin approval queue --
    # ticking is_verified here is what grants a coach account full privileges.
    list_display = ('academy_name', 'user', 'is_verified', 'created_at')
    list_filter = ('is_verified',)
    list_editable = ('is_verified',)
    search_fields = ('academy_name', 'user__email')
