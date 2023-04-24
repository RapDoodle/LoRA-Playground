$('.upload-file').on('hidden.bs.modal', function (event) {
    $("body").addClass("modal-open");
})
//mouse hold message show: title
$(function () {
    $('.CILOs-description').tooltip();
});
function descrption_update(obj) {
    var e = obj;
    $(e).attr("title", document.getElementById($(e).attr('id')).value)
}
function minus(obj) {
    var e = obj;
    $(e).parent().parent().remove();
}

var edit_related_CILOs_text = ""
function edit_set_new_course_Related_CILOs_value(obj) {
    var e = obj
    edit_related_CILOs_text = ""
    $(e).parents(".edit-Related-CILOs-dropmenu").find(":checked").each(function () {
        if (edit_related_CILOs_text === "") {
            edit_related_CILOs_text += + $(this).val()
        } else {
            edit_related_CILOs_text += "," + $(this).val()
        }
    })
    $(e).parents(".Related-CILOs-input").find(".Related-CILOs-input-area").attr("value", edit_related_CILOs_text)

}
var edit_cilo_index = 0;
var Related_cilos
function edit_read(obj) {
    edit_cilo_index = 0;
    var e = $(obj).parents(".Action-column")
    $("#edit-Course-name").attr("value", e.siblings(".Course-name").attr("value"))
    $("#edit-Course-name").text(e.siblings(".Course-name").text())
    $("#edit-Course-code").text(e.siblings(".Course-code").text())
    $("#edit-CILOs-table").find(".Edit-CILOs-tbale-tr").remove()
    $("#edit-CILOs-table").append(e.find(".course-hide-CILOs-table").html())
    $("#edit-CILOs-table").attr("value", e.find(".course-hide-CILOs-table").attr("id"))
    $("#edit-CILOs-table").find(".CILOs-description").each(function () {
        $(this).attr("id", "edit-cilo-index-" + edit_cilo_index)
        edit_cilo_index++
    })
    $("#edit-CILOs-table").find(".CILOs-description").each(function () {
        descrption_update($(this))
    })
    $("#edit-Assessment-methods-table").find(".Edit-assessment-methods-tbale-tr").remove()
    $("#edit-Assessment-methods-table").append(e.find(".course-hide-assessment-methods-table").html())
    $("#edit-Assessment-methods-table").attr("value", e.find(".course-hide-assessment-methods-table").attr("id"))
    edit_get_line_nums()
    $("#edit-Assessment-methods-table").find(".Related-CILOs-input").each(function () {
        Related_cilos = $(this).find(".Related-cilo-dropdown-button").val().split(",")
        $(this).find(".Related_CILOs_CheckBox").each(function () {
            for (var i = 0; i < Related_cilos.length; i++) {
                if ($(this).val() == Related_cilos[i]) {
                    $(this).attr("checked", "checked")
                    break;
                }
            }
        })
    })
}

function set_import_type(type) {
    $("#import-file-type").attr("value", type)
}
$(".custom-file-input").on("change", function () {
    var file_name = $(this).val().split("\\").pop();
    $(this).siblings(".custom-file-label").addClass("selected").html(file_name);
});

