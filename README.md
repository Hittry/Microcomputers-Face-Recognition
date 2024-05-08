# Microcomputers-Face-Recognition

### Модели
Все необходимы модели лежат в папке [dist](dist)

### Изолированное устройство
* [isolation-setup](isolation-setup) -- Код для работы сервиса распознавания на изолированном устройстве
* [get-descriptors](isolation-setup%2Fget-descriptors) -- Скрипт получения дескрипторов лиц для идентификации пользователей
* [recognition-service](isolation-setup%2Frecognition-service) -- Сервис распознавания лиц. В [dist](isolation-setup%2Frecognition-service%2Fdist) 
нужно добавить результаты работы скрипта получения дескрипторов лиц и необходимые модели.

Для изолированного устройства используются моедли:
- [centerface_scripted.pt](dist%2Fcenterface_scripted.pt);
- [face_recognition.pt](dist%2Fface_recognition.pt).

### Промежуточное устройство
* [filter-setup](filter-setup) -- Код для работы сервиса распознавания.
* [get-descriptors](filter-setup%2Fget-descriptors) -- Скрипт для получения дескрипторов лиц.
* [recognition-service](filter-setup%2Frecognition-service) -- Сервис распознавания лиц. В [dist](filter-setup%2Frecognition-service%2Fdist)
нужно добавить указанные ниже модели.

Для промежуточного устройства используются модели:
- [centerface_scripted.pt](dist%2Fcenterface_scripted.pt);
- [edge_face_s.scripted](dist%2Fedge_face_s.scripted)
