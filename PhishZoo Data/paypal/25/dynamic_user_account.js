
PAYPAL.namespace("Marketing");PAYPAL.Marketing.PayPalUsers={SELF_NAME:"PAYPAL.Marketing.PayPalUsers",DEFAULT_SEED_COUNT:219000000,DEFAULT_SEED_DATE:"May 26, 2010",DEFAULT_GROUP_SEPARATOR:",",DEFAULT_TARGET_ELEMENT_ID:"NoOfUserLogin",DEFAULT_UPDATE_INTERVAL:3000,DEFAULT_COUNT_INCREMENT:2,UserAccounts:0,UAObj:null,init:function(){this.UAObj=document.getElementById(this.DEFAULT_TARGET_ELEMENT_ID);if(this.UAObj){this.setInitialAccount();this.updateCount();}},formatNumber:function(number){nStr=number.toString();var rgx=/(\d+)(\d{3})/;while(rgx.test(nStr)){nStr=nStr.replace(rgx,'$1'+this.DEFAULT_GROUP_SEPARATOR+'$2');}
return nStr;},setInitialAccount:function(){now=new Date();referenceDate=new Date(this.DEFAULT_SEED_DATE);deltaMilliseconds=now-referenceDate;this.UserAccounts=Math.round(((deltaMilliseconds/this.DEFAULT_UPDATE_INTERVAL)*this.DEFAULT_COUNT_INCREMENT)+this.DEFAULT_SEED_COUNT);},updateCount:function(){this.UserAccounts+=this.DEFAULT_COUNT_INCREMENT;this.renderCount(this.formatNumber((this.UserAccounts).toString()));setTimeout(this.SELF_NAME+".updateCount()",this.DEFAULT_UPDATE_INTERVAL);},renderCount:function(count){this.UAObj.innerHTML=count;}}