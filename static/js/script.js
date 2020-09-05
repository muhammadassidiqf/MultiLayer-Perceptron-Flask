const iconMenu = document.getElementById("icon-menu");
const menuContainer = document.querySelector(".menus");
const menuList = document.querySelectorAll(".list");
const sectionsArray = document.querySelectorAll("section");
const sectionsPos = {};

iconMenu.addEventListener("click", () => {
  if (iconMenu.classList[1] == "fa-bars") {
    changeIcon("close", "fa-bars", "fa-times");
    handleMenuContainer("slideInLeft", "slideOutLeft");
    showMenus(true);
  } else {
    handleMenuContainer("slideOutLeft", "slideInLeft");
    showMenus(false);
    changeIcon("", "fa-times", "fa-bars");
    return;
  }
});

function handleMenuContainer(toAdd, toRemove) {
  menuContainer.classList.add("animated", "forwards", toAdd);
  menuContainer.classList.remove(toRemove);
}

function changeIcon(type, beReplaced, changeTo) {
  iconMenu.classList.replace(beReplaced, changeTo);
  if (type === "close") {
    iconMenu.classList.add("close");
  } else {
    iconMenu.classList.remove("close");
  }
}

function showMenus(visibility) {
  let i = 6;
  if (visibility) {
    for (const list of menuList) {
      list.classList.add(
        "animated",
        "forwards",
        "bouncedIn",
        "fast",
        "mydelay-0${i}"
      );
      i += 6;
    }
  } else {
    for (const list of menuList) {
      list.removeAttribute("class");
    }
  }
}

sectionsArray.forEach((section) => {
  sectionsPos[section.id] = section.offsetTop;
});

window.onscroll = () => {
  var scrollPosition = document.documentElement.scrollTop || document.body.scrollTop;
  for (const id in sectionsPos) {
    if (sectionsPos[id] <= scrollPosition) {
      document.querySelector(".active").classList.remove("active");
      document.querySelector(`a[href*=${id}]`).classList.add("active");
      handleMenuContainer("slideOutLeft", "slideInLeft");
      showMenus(false);
      changeIcon("", "fa-times", "fa-bars");
      return;
    }
  }
};
