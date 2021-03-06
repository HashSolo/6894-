//  utility javascript file... for common validation and error display

if (top.location != self.location)
{
        top.location = self.location;
}

var opt="toolbar=no,directories=no,location=no,status=no,scrollbars=yes,resizable=yes,copyhistory=no";
var opt480=opt + ",width=600,height=480";
var _error_on_screen = false;
var validationMessages = new Array();

function openHelp(url,custType)
{
    open(url, "helpwindow_" + custType, opt480);
}


function openWindow(url)
{

	open(url, "", "");
}

function add_error(_error)
{
    var el = validationMessages.length;
    var _already_in = false;
    for (n in validationMessages) {
        if( _error.indexOf( validationMessages[n]) > -1)
        {
            _already_in = true;
        }
    }

    if(_already_in == false)
    {
        validationMessages[el] = _error;
    }
}

function show_error_messages()
{
    _error_on_screen = false;
    if( validationMessages.length == 0 )
    {
        return;
    }
    else
    {
        _error_on_screen = true;
    }

    output = '<table width=100% border=0 cellspacing=0 cellpadding=1 summary="' + STR_THISTABLE + '" class=errorbrdr>'
        + '<tr><td>'
        + '<table width=100% border=0 cellspacing=0 cellpadding=7 summary="' + STR_THISTABLE + '" class=errorbrdr><tr>'
        + '<td width=55 valign=top style="background-color: #ffffff;" ><img src="sas-docs/images/icon_alert_error.gif" alt="' + STR_ERRORMESSAGE + '"></td>'
        + '<td valign=top style="background-color: #ffffff;"><font class="text1">'

    var error_messages = '<table border=0 cellspacing=0 cellpadding=0 summary="' + STR_THISTABLE + '">';
    for (n in validationMessages) {
       error_messages = error_messages
                    + '<tr>'
                    + '<td class="text1">' + (validationMessages[n])
                    + '<br><img src="sas-docs/images/clr.gif" width=1 height=4></td></tr>';
    }
    error_messages = error_messages + '</table>';
    output = output + error_messages;
    output = output + '</td>'
            +' </tr>'
            +' </table>'
            +' </td></tr></table>';

    display_error_messages();

    validationMessages = new Array();
}

function display_error_messages()
{
    if (document.getElementById) {
        if (window.HTMLElement) {
            eNode = document.getElementById('error_section');
            if( document.getElementById('server_error_section') )
                eNode = document.getElementById('server_error_section');
            while (eNode.hasChildNodes()) eNode.removeChild(eNode.lastChild);
            var range = document.createRange();
            range.selectNodeContents(eNode);
            eNode.appendChild(range.createContextualFragment(output));
        }
        else {
            if( document.all('server_error_section') )
                document.all('server_error_section').innerHTML = output;
            else
                document.all('error_section').innerHTML = output;
        }
    }
    else if (document.all) {
            if( document.all('server_error_section') )
                document.all('server_error_section').innerHTML = output;
            else
                document.all('error_section').innerHTML = output;
    }
}


/* Multiply the size of the font for each style sheet rule
   for all linked and embedded style sheets.
*/
function multipleFontSize(factor)
{
    var styleSheet;
    var i;
    var done;

    for(i=0;i<document.styleSheets.length;i++)
    {
        styleSheet = document.styleSheets[i].cssText;
        styleSheet = styleSheet.toLowerCase();

        var pattern = /font-size\s*:\s*([\d\.]+)((em)|%)+/g;
        pattern.multiline = true;
        var result;
        done = false;

        while (!done)
        {
            var result = pattern.exec(styleSheet);
            if (result == null)
            {
                done = true;
            }
            else
            {
                strLeft = styleSheet.substring(0, result.index-1);
                strMid = result[0];
                strRight = styleSheet.substring(result.index +  result[0].length);
                size = result[1];
                size *= factor;
                var number_pattern = /(\d*.?\d{0,2})\d*/;
                var number_result = number_pattern.exec(size);
                strMid = strMid.replace(result[1], number_result[1]);
                styleSheet = strLeft + strMid + strRight;
            }
        }
        document.styleSheets[i].cssText= styleSheet;
    }
}

/* Examine the default page font.  If too small,
   increase by a percent factor
*/
function examineFontSize(ref)
{
    if (document.getElementById)
    {
        if (document.getElementById(ref) )
        {
            if (document.getElementById(ref).currentStyle)
            {
                var size = document.getElementById(ref).currentStyle.fontSize;
                var index = size.indexOf("pt");
                var newstr = parseInt(size.substr(0,index));

                if (newstr < 10)
                {
                    multipleFontSize(1.3);
                }
                else if (newstr < 12)
                {
                    multipleFontSize(1.2);
                }
            }
        }
    }
}

//this code exists to extract locale from the image path passed into the function for non-locale sensitive
//images.  the LOCALE variable exists in noTranslate.properties.js for each locale so that in
//the event that more languages are added to the platform, this function will not need modified.
function getNonLocaleImagePath(baseURL) {
	var nonLocaleBaseURL = "";
	if (baseURL.search(LOCALE) != "-1") {
		var localePosition = baseURL.search(LOCALE);
		var baseURLStart = baseURL.slice(0, localePosition);
		var baseURLEnd = baseURL.slice(localePosition + 6);
		nonLocaleBaseURL = baseURLStart + baseURLEnd
		return nonLocaleBaseURL;
	}
}

function create_button(btnText, btnHref, css_class, onclick_evt, onmouseover_evt, onmouseout_evt, tabindex)
{
	getButton(btnText, btnHref, "", onclick_evt, "sas-docs/images/", btnText,  css_class, tabindex)
}

function getButton(btnText, btnHref, btnTarget, btnOnClick, baseURL, btnTitle,  css_class, tabindex)
{
	//get baseURL without locale
	var nonLocaleBaseURL = getNonLocaleImagePath(baseURL)

	/// Note:  an extra, optional argument may be passed to this function
	/// which specifies the window that should be written to.  If the
	/// argument is not supplied, the current document window is used.
	if (getButton.arguments.length > 8)
	{
		doc = getButton.arguments[8];
	}
	else
	{
		doc = document;
	}

	var t = "";

	t = "<table border=0 cellpadding=0 cellspacing=0 summary=\"\" class=\"" + css_class + "\"><tr><td width=2 rowspan=2 class=\"" + css_class + "-left\"><img src='" + nonLocaleBaseURL + "clr.gif' alt=\"\" width=2 height=2><\/td><td class=\"" + css_class + "-top\"><img src='" + baseURL + "clr.gif' alt=\"\" width=1 height=2><\/td><td width=2 rowspan=3 class=\"" + css_class + "-right\"><img src='" + nonLocaleBaseURL + "clr.gif' alt=\"\" width=2 height=2><\/td><\/tr><tr><td nowrap><div class=\"" + css_class + "\" ";
	t = t + "><a href=\"" + btnHref + "\" target=\"" + btnTarget + "\" class=\"" + css_class  + "\" "
		+ " onFocus='hover(this, \"" + css_class + "-over\");window.status=\"" + btnTitle + "\";'"
		+ " onBlur='hover(this, \"" + css_class + "\");window.status=\"\";'"
		+ " title=\"" + btnTitle + "\" ";
	if (btnOnClick) { t = t + " \'" + btnOnClick + "\'"; }
	if (tabindex) { t = t + " tabindex=\"" + tabindex + "\""; }

	t = t + ">&nbsp;" + btnText + "&nbsp;<\/a><\/div><\/td><\/tr><tr><td colspan=2 class=\"" + css_class +  "-right\"><img src='" + nonLocaleBaseURL + "clr.gif' alt=\"\" width=1 height=2><\/td><\/tr><\/table>";

	doc.write(t);
}

function getTwoButtons(btnText, btnHref, btnTarget, btnOnClick, baseURL, btnTitle,  css_class, tabindex,
						btnText2, btnHref2, btnTarget2, btnOnClick2, baseURL2, btnTitle2,  css_class2, tabindex2)
{
	//get baseURL without locale
	var nonLocaleBaseURL = getNonLocaleImagePath(baseURL)

	/// Note:  an extra, optional argument may be passed to this function
	/// which specifies the window that should be written to.  If the
	/// argument is not supplied, the current document window is used.
	if (getTwoButtons.arguments.length > 16)
	{
		doc = getTwoButtons.arguments[16];
	}
	else
	{
		doc = document;
	}

	var t = "";

	t = "<table border=0 cellpadding=0 cellspacing=0 summary=\"\" class=\"" + css_class + "\"><tr><td width=2 rowspan=2 class=\"" + css_class + "-left\"><img src='" + nonLocaleBaseURL + "clr.gif' alt=\"\" width=2 height=2><\/td><td width=2 rowspan=2 class=\"" + css_class2 + "-left\"><img src='" + nonLocaleBaseURL + "clr.gif' alt=\"\" width=2 height=2><\/td><td class=\"" + css_class + "-top\"><img src='" + baseURL + "clr.gif' alt=\"\" width=1 height=2><\/td><td class=\"" + css_class2 + "-top\"><img src='" + baseURL2 + "clr.gif' alt=\"\" width=1 height=2><\/td><td width=2 rowspan=3 class=\"" + css_class + "-right\"><img src='" + nonLocaleBaseURL + "clr.gif' alt=\"\" width=2 height=2><\/td><td width=2 rowspan=3 class=\"" + css_class2 + "-right\"><img src='" + nonLocaleBaseURL + "clr.gif' alt=\"\" width=2 height=2><\/td><\/tr><tr><td nowrap><div class=\"" + css_class + "\" ";
	t = t + "><a href=\"" + btnHref + "\" target=\"" + btnTarget + "\" class=\"" + css_class  + "\" "
		+ " onFocus='hover(this, \"" + css_class + "-over\");window.status=\"" + btnTitle + "\";'"
		+ " onBlur='hover(this, \"" + css_class + "\");window.status=\"\";'"
		+ " title=\"" + btnTitle + "\" ";


	t = t + ">&nbsp;" + btnText + "&nbsp;<\/a><\/div><\/td><td nowrap><div class=\"" + css_class2 + "\" ";
			
		t = t + "><a href=\"" + btnHref2 + "\" target=\"" + btnTarget2 + "\" class=\"" + css_class2  + "\" "
		+ " onFocus='hover(this, \"" + css_class2 + "-over\");window.status=\"" + btnTitle2 + "\";'"
		+ " onBlur='hover(this, \"" + css_class2 + "\");window.status=\"\";'"
		+ " title=\"" + btnTitle2 + "\" ";
		
		
	t = t + ">&nbsp;" + btnText2 + "&nbsp;<\/a><\/div><\/td><\/tr><tr><td colspan=2 class=\"" + css_class +  "-right\"><img src='" + nonLocaleBaseURL + "clr.gif' alt=\"\" width=1 height=2><\/td><td colspan=2 class=\"" + css_class2 +  "-right\"><img src='" + nonLocaleBaseURL + "clr.gif' alt=\"\" width=1 height=2><\/td><\/tr><\/table>";

	doc.write(t);
	
}


function hover(ref, classRef)
{
	eval(ref).className = classRef;
}

function checkForCookies()
{
	var today = new Date();
	var expire = new Date();
	expire.setTime(today.getTime() + 3600000 * 24);
	document.cookie = "cookieEnabledPersist=true; expires=" + expire.toGMTString();
	document.cookie = "cookieEnabledSession=true";
	var cookieString = ""+document.cookie;
	if (cookieString.indexOf("cookieEnabledPersist")==-1 || cookieString.indexOf("cookieEnabledSession")==-1)
	{
		document.location.href = "noCookies.do";
	}
}

function loadGating()
{
	checkForCookies();
}

function displayDynamicText(output_text)
{
    if( document.getElementById )
    {
        document.getElementById("dynamicMsg").innerHTML = output_text;
    }
    else if( document.all )
    {
        document.all.dynamicMsg.innerHTML = output_text;
    }
    return;
}

// set the correct style sheet for the browser

var platform= navigator.platform.toLowerCase(); // Operating system
var appName = navigator.appName;                // Browser name
var appVer= parseInt(navigator.appVersion);     // Browser versionnumber

function checkBrowser(doc)
{
	doc.writeln('<link rel="stylesheet" type="text/css" href="' + GLOBAL_CSS + '">');
}


//Image functions
//this  is to cache all the images required at the load time.
var myimages=new Array()
var gotolink="#"

//StrongSecurity Start
//Image functions
function preloadimages() {
	preloadImages();
}

function preloadImages() {
  var d=document; if(d.images){ if(!d.MM_p) d.MM_p=new Array();
    var i,j=d.MM_p.length,a=preloadImages.arguments; for(i=0; i<a.length; i++)
    if (a[i].indexOf("#")!=0){ d.MM_p[j]=new Image; d.MM_p[j++].src="/images/"+a[i];}}
}

function swapImgRestore() {
  var i,x,a=document.MM_sr; for(i=0;a&&i<a.length&&(x=a[i])&&x.oSrc;i++) x.src=x.oSrc;
}

function findObj(n, d) {
  var p,i,x;  if(!d) d=document; if((p=n.indexOf("?"))>0&&parent.frames.length) {
    d=parent.frames[n.substring(p+1)].document; n=n.substring(0,p);}
  if(!(x=d[n])&&d.all) x=d.all[n]; for (i=0;!x&&i<d.forms.length;i++) x=d.forms[i][n];
  for(i=0;!x&&d.layers&&i<d.layers.length;i++) x=findObj(n,d.layers[i].document);
  if(!x && document.getElementById) x=document.getElementById(n); return x;
}

function swapImage() {
  var i,j=0,x,a=swapImage.arguments; document.MM_sr=new Array; for(i=0;i<(a.length-2);i+=3)
   if ((x=findObj(a[i]))!=null){document.MM_sr[j++]=x; if(!x.oSrc) x.oSrc=x.src; x.src=a[i+2];}
}

//Get image button
function getImageButton(btnText, btnHref, btnTarget, btnOnClick, baseURL, btnTitle,  css_class, tabindex , linkId, imageName,
	regular_image, onmouseover_image, onmousedown_image, width, height, border)
{
  	//get baseURL without locale
	var nonLocaleBaseURL = getNonLocaleImagePath(baseURL)

	/// Note:  an extra, optional argument may be passed to this function
    /// which specifies the window that should be written to.  If the
    /// argument is not supplied, the current document window is used.
    if(getImageButton.arguments.length > 16) {
		doc = getImageButton.arguments[16];
    }
    else {
		doc = document;
    }
    var t = "";
    if (( doc.getElementById )||( doc.all )) {
		t = "<table border=0 cellpadding=0 cellspacing=0";

		if ( width ) {
			t = t + " width=\"" + width + "\"";
		}

		t = t + " summary=\"\" class=" + css_class
			+ "><tr><td ";

		if ( width ) {
			t = t + " width=\"" + width + "\"";
		}

		t = t + "><div class=" + css_class
			+ "><a href=\"" + btnHref
			+ "\" target=\"" + btnTarget;

		//MSIE 5.01 doesn't support css style within this method.
		//css class has to be commented out
		//if ( navigator.userAgent.indexOf("MSIE 5.01") == -1 ) {
		//	t = t + "\" class=\"" + css_class + "\"";
		//}

		t = t + "\" id=\"" + linkId
			+ "\"><img "
			+ "name=\"" + imageName
			+ "\" src=\"" + regular_image + "\"";

		if(onmouseover_image) {
			t = t + " onMouseOver=\"swapImage('" + imageName + "','','" + onmouseover_image  + "',1);\"";
		}

		if(onmouseover_image || onmousedown_image) {
			t = t + " onMouseOut=\"swapImgRestore();\"";
		}

		if(onmousedown_image) {
			t = t + " onMouseDown=\"swapImage('" + imageName + "','','" + onmousedown_image  + "',1);\"";
		}

		if (btnTitle) { t = t + " title=\"" + btnTitle + "\""; }
		if (btnOnClick) { t = t + " " + btnOnClick + " "; }
		if (tabindex) { t = t + " tabindex=\"" + tabindex + "\""; }
		if ( width ) {
			t = t + " width=\"" + width + "\"";
		}
		if ( height ) {
			t = t + " height=\"" + height + "\"";
		}
		if ( border ) {
			t = t + " border=\"" + border + "\"";
		} else {
			t = t + " border=\"0\"";
		}
		if ( btnTitle ) {
			t = t + " alt=\"" + btnTitle + "\"";
		}

		t = t + "><\/a><\/div><\/td><\/tr><\/table>";
	}
	else {
		t = "<table border=0 cellpadding=0 cellspacing=0 summary=\"\" class=" + css_class + "><tr><td width=2 rowspan=2 class="
		+ css_class + "-left><img src='" + nonLocaleBaseURL + "clr.gif' alt=\"\" width=2 height=2><\/td><td class=" + css_class + "-top><img src='" + nonLocaleBaseURL + "clr.gif' alt=\"\" width=1 height=2><\/td><td width=2 rowspan=3 class=" + css_class + "-right><img src='" + nonLocaleBaseURL + "clr.gif' alt=\"\" width=2 height=2><\/td><\/tr><tr><td><div class=" + css_class + "style=\"padding: 1px 3px 1px 3px;\">"
		+ "<a href=\"" + btnHref + "\" target=\"" + btnTarget + "\" id=\"" + linkId + "\""

		if (btnTitle) { t = t + " title=\"" + btnTitle + "\""; }
		if (btnOnClick) { t = t + " " + btnOnClick + " "; }
		if (tabindex) { t = t + " tabindex=\"" + tabindex + "\""; }

		t = t + "><img name=\"" + imageName + "\" src=\"" + regular_image + "\"";

		if ( width ) {
			t = t + " width=\"" + width + "\"";
		}
		if ( height ) {
			t = t + " height=\"" + height + "\"";
		}
		if ( border ) {
			t = t + " border=\"" + border + "\"";
		} else {
			t = t + " border=\"0\"";
		}
		if ( btnTitle ) {
			t = t + " alt=\"" + btnTitle + "\"";
		}

		t = t + "><\/a><\/div><\/td><\/tr><tr><td colspan=2 class=" + css_class +  "-right><img src='" + nonLocaleBaseURL + "clr.gif' alt=\"\" width=1 height=2><\/td><\/tr><\/table>";
	}

	//if ( debug ) {
	//	logMsg(t);
	//}

	doc.write(t);
}

function getImageAnchor(btnText, btnHref, btnTarget, btnOnClick, baseURL, btnTitle, ancText, ancClass,
	linkId, imageName, imageSrc, width, height, border)
{
	var t = "<a href=\"" + btnHref + "\" target=\"" + btnTarget + "\" title=\"" + btnTitle + "\" id=\"" + linkId +  "\" class=" + ancClass + ">"
		+ "<img src=\"" + imageSrc + "\" name=\"" + imageName + "\"";


	if ( width ) {
		t = t + " width=\"" + width + "\"";
	}
	if ( height ) {
		t = t + " height=\"" + height + "\"";
	}
	if ( border ) {
		t = t + " border=\"" + border + "\"";
	} else {
		t = t + " border=\"0\"";
	}
	if ( btnTitle ) {
		t = t + " alt=\"" + btnTitle + "\"";
	}

	t = t + "><\/a>";

	document.write(t);
}


function disableImageButton(linkId, imageName, disabledImage) {

	var linkObj = document.getElementById(linkId);
	var imageObj = document[imageName];

	if ( linkObj != null && (linkObj.getAttribute("btnDisabled") == null || linkObj.getAttribute("btnDisabled") != "true" ) ) {
		var href = linkObj.getAttribute("href");
		var onclick = linkObj.getAttribute("onclick");
		var title = linkObj.getAttribute("title");

		linkObj.setAttribute("href_bak", href);
		linkObj.setAttribute("onclick_bak", onclick);
		linkObj.setAttribute("title_bak", title);

		linkObj.setAttribute("href", "");
		linkObj.setAttribute("onclick", "");
		linkObj.setAttribute("title", "");

		linkObj.removeAttribute("href");
		linkObj.removeAttribute("onclick");
		linkObj.removeAttribute("title");

		linkObj.setAttribute("btnDisabled", "true");
		linkObj.disabled = true;
	}

	if (imageObj != null && (imageObj.getAttribute("btnDisabled") == null || imageObj.getAttribute("btnDisabled") != "true" )) {
		var src = imageObj.src;
		var onmouseover = imageObj.getAttribute("onmouseover");
		var onmouseout = imageObj.getAttribute("onmouseout");
		var onmousedown = imageObj.getAttribute("onmousedown");
		var title = imageObj.getAttribute("title");

		imageObj.setAttribute("src_bak", src);
		imageObj.setAttribute("onmouseover_bak", onmouseover);
		imageObj.setAttribute("onmouseout_bak", onmouseout);
		imageObj.setAttribute("onmousedown_bak", onmousedown);
		imageObj.setAttribute("title_bak", title);

		imageObj.setAttribute("src", disabledImage);
		imageObj.setAttribute("onmouseover", null);
		imageObj.setAttribute("onmouseout", null);
		imageObj.setAttribute("onmousedown", null);
		imageObj.setAttribute("title", "");

		imageObj.removeAttribute("onmouseover");
		imageObj.removeAttribute("onmouseout");
		imageObj.removeAttribute("onmousedown");
		imageObj.removeAttribute("title");

		imageObj.setAttribute("btnDisabled", "true");
		imageObj.disabled = true;
	}
}

function enableImageButton(linkId, imageName, enabledImage) {
	var linkObj = document.getElementById(linkId);
	var imageObj = document[imageName];

	if ( linkObj!= null && (linkObj.getAttribute("btnDisabled") != null && linkObj.getAttribute("btnDisabled") == "true" )) {

		linkObj.setAttribute("href", linkObj.getAttribute("href_bak"));
		linkObj.setAttribute("onclick", linkObj.getAttribute("onclick_bak"));
		linkObj.setAttribute("title", linkObj.getAttribute("title_bak"));
		linkObj.setAttribute("btnDisabled", "false");
		linkObj.disabled = false;
	}

	if (imageObj != null && (imageObj.getAttribute("btnDisabled") != null && imageObj.getAttribute("btnDisabled") == "true" )) {
		imageObj.setAttribute("src", imageObj.getAttribute("src_bak"));

		imageObj.setAttribute("onmouseover", imageObj.getAttribute("onmouseover_bak"));
		imageObj.setAttribute("onmouseout", imageObj.getAttribute("onmouseout_bak"));
		imageObj.setAttribute("onmousedown", imageObj.getAttribute("onmousedown_bak"));
		imageObj.setAttribute("title", imageObj.getAttribute("title_bak"));

		imageObj.setAttribute("btnDisabled", "false");
		imageObj.disabled = false;
	}
}

//Toggle language handler
function languageToggleHandler(e) {
	//for firefox event hanbdling
	if (!e) var e = window.event;

	var langToggle = document.getElementById('langToggleM');
	var menuToggle = document.getElementById('langMenuM');
	var langPrefLink = document.getElementById('en-langPref');
	var langPrefLinkEs = document.getElementById('es-langPref');
	var arrow = document.getElementById('arrowIcon');

	//show menu
	//toggle "Show language preference menu" to "Hide language preference menu"
	if (e.type == "mouseover") {
		setCSSClass(langToggle.id,"on");
		setCSSClass(menuToggle.id,"displayLangLincs");
	}

	if (e.type == "mouseout") {
		setCSSClass(langToggle.id,"");
		setCSSClass(menuToggle.id,"");
	}

	//show menu
	//toggle "Show language preference menu" to "Hide language preference menu"
	if (e.type == "focus") {
		setCSSClass(langToggle.id,"on");
		setCSSClass(menuToggle.id,"displayLangLincs");
		arrow.focus();
	}

	//hide menu
	//toggle "Hide language preference menu" to "Show language preference menu"
	if (e.type == "blur") {
		if (focusIndicator == langPrefLink.id || focusIndicator == langPrefLinkEs.id) {
			var focusVar = document.getElementById(focusIndicator);
			focusVar.focus();
		} else {
			setCSSClass(langToggle.id,"");
			setCSSClass(menuToggle.id,"");
		}
	}

	if (e.type == "keydown") {
	}
}

function navigateMenu(e) {
	if (!e) var e = window.event;
	var increment = 0;
	var focusElement;

	var langPrefLink = document.getElementById('en-langPref');
	var langPrefLinkEs = document.getElementById('es-langPref');
	var arrow = document.getElementById('arrowIcon');

	switch (e.keyCode) {

		case 38:
			if (focusIndicator == langPrefLinkEs.id) {
				focusIndicator = langPrefLink.id;
				langPrefLink.focus();
			} else if (focusIndicator == langPrefLink.id) {
				arrow.focus();
				focusIndicator = "";
			}
		break
		case 40:
			if (focusIndicator == "") {
				focusIndicator = langPrefLink.id;
				langPrefLink.focus();
			} else if (focusIndicator == langPrefLink.id) {
				focusIndicator = langPrefLinkEs.id;
				langPrefLinkEs.focus();
			}
		break
	}
}

function openOutOfLanguagePopup(continueURL) {
    var url = "outOfLanguagePopupScreen.do?continueURL=" + continueURL;
    var win = window.open(url, 'outOfLanguagePopup','width=500,height=350,scrollbars=no,resizable=0');
    if (win.opener == null) win.opener = self;
}
