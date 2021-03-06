/* ========================================================================================

	File Name:	dhtml.js
	Purpose:	This is a generic script file for performing DHTML
                please use this file as a baseline for any other DHTML scripts
                
	Developer: 	Bill Quinlan Bank of America
	Copyright:	April 2005
   	Modifications:
	  Date			Developer	Change
	-------------------------------------------------------
	April 		2005	Bill Quinlan	Genisis
	
	November	2006 	Erin Riggers 	Major modifications
			* Removal of support for versions of Netscape prior to 6.x (no more support for document.layers)
			* Tempted to remove support for IE vervions previous to 5.5 and Mac 5.2, 
			  since we don't support them (no more support for document.all)
			* New DHTML functions
				- toggle display
				- get class
				- set class
				- toggle class
				
	February	2007	Erin Riggers	Enhancements for Ledger Balance project
   ========================================================================================   
*/	
var dom = document.getElementById? 1 : 0;
var ie  = document.all? 1 : 0;


/*** hide/show functions ***/
function showObject(objList){
	var a = objList.split(',');
	for(var i = 0; i < a.length; i++)
	{
		show(a[i]);
	}
}

function hideObject(objList){
	var a = objList.split(',');
	for(var i = 0; i < a.length; i++)
	{
		hide(a[i]);
	}
}

function toggleObject(objList) {
	var a = objList.split(',');
	for(var i = 0; i < a.length; i++)
	{
		toggle(a[i]);
	}
}

function show(obj)
{
	obj = findObj(obj);
	if(!obj) return;
	obj.style.visibility = "visible";
	obj.style.display = "";
}

function hide(obj){
	obj = findObj(obj);
	if(!obj) return;
	obj.style.visibility = "hidden";
	obj.style.display = "none";
}

function toggle(elementId) {
	var obj = findObj(elementId);
	if(!obj) return;
	if (obj.style.display == "none") {
		show(elementId);
	} else {
		hide(elementId);
	}
}

/*** get/set/toggle CSS class functions ***/

//get CSS class in use for a specific element
function getCSSClass(obj) {
	var obj = findObj(obj);
	if(!obj) return;
	
	return obj.className;
}

//set class for given element
//speficy element name and name of class to change to
function setCSSClass(obj,className) {
	var obj = findObj(obj);
	if(!obj) return;
	obj.className = className;
}

//toggle between CSS classes.  The on state class is expected to
//be named as baseClass + "On"
function toggleClass(obj, baseClass, onClass) {
	var obj = findObj(obj);
	if(!obj) return;
	
	if (obj.className == baseClass) {
		obj.className = onClass;
	} else {
		obj.className = baseClass;
	}
}


/* Change Cursor Function */
function swapCursor(cursorStyle) {
	document.body.style.cursor = cursorStyle;	
}


function returnObj(obj) {

}

function findObj(obj){
	var p, i, found, doc;
	doc = document;
	
	if(!(found = doc[obj]) && ie){
		found = doc.all[obj];
	}
	for (i=0; !found && i < doc.forms.length; i++){
		found = doc.forms[i][obj];
	}
	if(!found && dom){
		found = document.getElementById(obj);
	}
	return found;
}

//This is a testing function which allows you to see what DOM your browser uses
//Use alert(isDHTML()); 
function isDHTML()
{
	var DHTML = (document.getElementById || document.all || document.layers);
	return DHTML;
}

