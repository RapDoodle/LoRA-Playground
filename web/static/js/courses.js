
//fix the problem of modal
$('.define-box').on('hidden.bs.modal', function (event) {
	$("body").addClass("modal-open");
})
//cilos define read function
function read_CILOs(obj) {
	var e = obj
	var id_text = $(e).parent().find(".CILOs-hide-table").attr("id");
	var table_concent = $(e).parent().find(".CILOs-hide-table").html();
	$(".define-box-added-table").children().remove();
	$(".define-box-added-table").attr("value", id_text);
	$(".define-box-added-table").append(table_concent);
}
function write_CILOs() {
	var id = "#" + $(".define-box-added-table").attr("value");
	var update_content = $(".define-box-added-table").html();
	$(id).children().remove();
	$(id).append(update_content);
	update_Pre_requisites();
}
function CILOs_define_page_minus(obj) {
	var e = obj;
	var e2 = $(e).parents(".CLIOs-define-tr")
	var plus = '<button type="button" class="btn btn-light" onclick="CILOs_define_page_add(this)">\
										<i class="fa fa-plus"></i>\
									</button>'
	$(e).parents(".CILOs-button-td").html(plus);
	var content = $(e2).html()
	var content_text = '<tr class="CLIOs-define-tr" style="text-align: center;">' + content + '</tr>'
	$(".define-box-result-table").append(content_text);
	$(e2).remove();
}

var Pre_requisites_list
function update_Pre_requisites() {
	Pre_requisites_list = new Array();
	$("#CILOs-table").find(".course-name-td").each(function () {

		Pre_requisites_list.push($(this).text())
	})
	var temp_array = new Array();
	for (var i = 0; i < Pre_requisites_list.length; i++) {
		var flag = true;
		for (var j = 0; j < temp_array.length; j++) {
			if (temp_array[j] === Pre_requisites_list[i]) {
				flag = false;
				break;
			}
		}
		if (flag) {
			temp_array.push(Pre_requisites_list[i]);
		}
	}
	$("#Pre-requisites-table").find(".Pre-requisites-tr").remove();
	for (var i = 0; i < temp_array.length; i++) {
		$("#Pre-requisites-table").append('<tr class="Pre-requisites-tr" style="text-align: center;"><td>' + temp_array[i] + "</tr></td>")
	}
}

function term_select(obj) {
	var e = obj;
	$("#new-course-term").attr("value", $(e).text());
}

function edit_term_select(obj) {
	var e = obj;
	$("#edit-effective-course-term").attr("value", $(e).text());
}

function program_select(obj) {
	var e = obj;
	var text = $(e).text();
	$("#new-course-program").attr("value", text);
	$("#new-course-program").attr("aria-label", $(e).attr("value"));
}
function course_type_select(obj) {
	var e = obj;
	var text = $(e).text();
	$("#new-course-type").attr("value", text);
	$("#new-course-type").attr("aria-label", $(e).attr("value"));
}
function CILOs_remove(obj) {
	minus(obj);
	update_CILOs_table();
	get_line_nums();
	$(".Related-CILOs-input-area").val("")
}
function set_search_type(obj) {
	$("#search-type-switch").attr("value", $(obj).attr("value"))
	$("#search-type-switch").text($(obj).text())
}
function update_CILOs_table() {
	let CILOsIndex = 1
	$("#CILOs-table").find("td.CILO-order").html(function () {
		var text = "CILO-" + CILOsIndex
		$(this).attr("value", CILOsIndex)
		CILOsIndex++
		return text
	})
	CILOsIndex = 1
	$("#CILOs-table").find("input.CILOs-description").attr("id", () => {
		var text = "CILOs-description-" + CILOsIndex
		CILOsIndex++
		return text
	})
}
var related_CILOs_text = ""
function set_new_course_Related_CILOs_value(obj) {
	var e = obj
	related_CILOs_text = ""
	$(e).parents(".Related-CILOs-dropmenu").find(":checked").each(function () {
		if (related_CILOs_text === "") {
			related_CILOs_text += + $(this).val()
		} else {
			related_CILOs_text += "," + $(this).val()
		}
	})
	$(e).parents(".Related-CILOs-input").find(".Related-CILOs-input-area").attr("value", related_CILOs_text)
}
//this function useless
// function edit_write() {
// }