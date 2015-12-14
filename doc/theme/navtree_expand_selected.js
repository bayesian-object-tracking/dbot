// Overloaded to remove links to sections/subsections
function getNode(o, po)
{
  po.childrenVisited = true;
  var l = po.childrenData.length-1;
  for (var i in po.childrenData) {
    var nodeData = po.childrenData[i];
    if((!nodeData[1]) ||  (nodeData[1].indexOf('#')==-1)) // <- we added this line
      po.children[i] = newNode(o, po, nodeData[0], nodeData[1], nodeData[2], i==l);
  }
}

function selectAndHighlight(hash,n)
{
  var a;
  if (hash) {
    var link=stripPath($(location).attr('pathname'))+':'+hash.substring(1);
    a=$('.item a[class$="'+link+'"]');
  }
  if (a && a.length) {
    a.parent().parent().addClass('selected');
    a.parent().parent().attr('id','selected');
    highlightAnchor();
  } else if (n) {
    $(n.itemDiv).addClass('selected');
    $(n.itemDiv).attr('id','selected');
  }
  if ($('#nav-tree-contents .item:first').hasClass('selected')) {
    $('#nav-sync').css('top','30px');
  } else {
    $('#nav-sync').css('top','5px');
  }
  expandNode(new Object(), n, true, true);
  showRoot();
};


// return false if the the node has no children at all, or has only section/subsection children
function checkChildrenData(node) {
  if (!(typeof(node.childrenData)==='string')) {
    for (var i in node.childrenData) {
      var url = node.childrenData[i][1];
      if(url.indexOf("#")==-1)
        return true;
    }
    return false;
  }
  return (node.childrenData);
};

// Modified to:
// - remove the section/subsection children
function createIndent(o,domNode,node,level)
{
  var level=-1;
  var n = node;
  while (n.parentNode) { level++; n=n.parentNode; }
  var imgNode = document.createElement("img");
  imgNode.style.paddingLeft=(16*(level)).toString()+'px';
  imgNode.width  = 16;
  imgNode.height = 22;
  imgNode.border = 0;
  if (checkChildrenData(node)) { // <- we modified this line to use checkChildrenData(node) instead of node.childrenData
    node.plus_img = imgNode;
    node.expandToggle = document.createElement("a");
    node.expandToggle.href = "javascript:void(0)";
    node.expandToggle.onclick = function() {
      if (node.expanded) {
        $(node.getChildrenUL()).slideUp("fast");
        node.plus_img.src = node.relpath+"ftv2pnode.png";
        node.expanded = false;
      } else {
        expandNode(o, node, false, false);
      }
    }
    node.expandToggle.appendChild(imgNode);
    domNode.appendChild(node.expandToggle);
    imgNode.src = node.relpath+"ftv2pnode.png";
  } else {
    imgNode.src = node.relpath+"ftv2node.png";
    domNode.appendChild(imgNode);
  }
};
