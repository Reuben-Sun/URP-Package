# URP-Package

> 基于com.unity.render-pipelines.universal@14.0.10，Work for [ToonURP](https://github.com/Reuben-Sun/ToonURP)

## 前言

当我想基于Unity URP做一个非侵入式的卡渲库，结果处处受阻

- 我想关掉内置的Shadow，自行实现，发现这个代码写死在URP Renderer中
- 我想实现一个Tonemapping，结果不改URP代码，很难在基础上拓展

我看了很多开源项目，他们都喜欢将URP完整拷贝一份，然后在这基础上进行修改，我感觉这好脏，但现在也不得不做了

所有对URP源码的修改，我都会在此记录
